from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from src.algorithms import DANN, ERM, MEIIS, MEIISConfig


class _DummyFeaturizer(torch.nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(int(x.shape[0]), int(self.feature_dim), device=x.device, dtype=torch.float32)


def test_feature_extraction_shape_consistent_across_methods() -> None:
    feature_dim = 7
    featurizer = _DummyFeaturizer(feature_dim)
    x = torch.randn(5, 3, 8, 8)

    erm = ERM(featurizer=featurizer, feature_dim=feature_dim, num_classes=2)
    dann = DANN(featurizer=featurizer, feature_dim=feature_dim, num_classes=2)
    meiis = MEIIS(featurizer=featurizer, feature_dim=feature_dim, num_classes=2, seed=0, config=MEIISConfig(K=1))

    f_erm = erm.extract_features(x)
    f_dann = dann.extract_features(x)
    f_meiis = meiis.extract_features(x)

    assert f_erm.shape == f_dann.shape == f_meiis.shape == (5, feature_dim)


class _SourceWithIndex(torch.utils.data.Dataset):
    def __init__(self, n: int = 4):
        self.n = int(n)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.zeros(3, 8, 8) + float(idx)
        y = torch.tensor(idx % 2, dtype=torch.long)
        metadata = torch.zeros(1, dtype=torch.long)
        return x, y, metadata, idx


def _make_target_batch(logits: torch.Tensor, *, y: Any = None) -> tuple[torch.Tensor, Any, torch.Tensor]:
    # x is unused by the patched forward(); still must have correct batch size.
    x = torch.zeros(int(logits.shape[0]), 3, 8, 8, dtype=torch.float32)
    metadata = torch.zeros(int(logits.shape[0]), 1, dtype=torch.long)
    return x, y, metadata


def test_me_iis_confidence_filtering(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MEIISConfig(
        K=1,
        use_confidence_filtered_constraints=True,
        target_conf_mode="maxprob",
        target_conf_thresh=0.90,
        target_conf_min_count=1,
        ema_constraints=0.0,
        weight_clip_max=10.0,
        weight_mix_alpha=1.0,  # disable mixing for easier expectation
    )
    meiis = MEIIS(featurizer=_DummyFeaturizer(4), feature_dim=4, num_classes=2, seed=0, config=cfg)

    # 4 target samples: two high-confidence, two low-confidence.
    logits = torch.tensor(
        [
            [10.0, 0.0],  # high conf class 0
            [0.0, 10.0],  # high conf class 1
            [0.2, 0.0],   # low conf
            [0.0, 0.2],   # low conf
        ],
        dtype=torch.float32,
    )
    target_loader = [_make_target_batch(logits)]
    source_loader = torch.utils.data.DataLoader(_SourceWithIndex(4), batch_size=2, shuffle=False)

    def _fake_forward(_x: torch.Tensor) -> torch.Tensor:
        return logits

    monkeypatch.setattr(meiis, "forward", _fake_forward)
    monkeypatch.setattr(meiis.adapter, "fit_target_structure", lambda *args, **kwargs: None)

    def _fake_get_joint_features(layer_features: Dict[str, torch.Tensor], class_probs: torch.Tensor):  # noqa: ANN001
        # J=1 so the joint flatten is just the class probs.
        probs = torch.as_tensor(class_probs, dtype=torch.float32)
        return {meiis.layer_name: probs.unsqueeze(1)}

    monkeypatch.setattr(meiis.adapter, "get_joint_features", _fake_get_joint_features)

    captured: Dict[str, Any] = {}

    def _fake_solve_iis_from_joint(*, target_moments_override: torch.Tensor, source_joint: Any, target_joint: Any, **kwargs):  # noqa: ANN001
        captured["target_moments_override"] = target_moments_override.detach().cpu()
        n_source = int(next(iter(source_joint.values())).shape[0])
        w = torch.full((n_source,), 1.0 / float(n_source), dtype=torch.float64)
        return w, 0.0, []

    monkeypatch.setattr(meiis.adapter, "solve_iis_from_joint", _fake_solve_iis_from_joint)

    out = meiis.update_importance_weights(source_loader=source_loader, target_loader=target_loader, device=torch.device("cpu"))
    assert out["status"] == "updated"
    assert out["target_total"] == 4
    assert out["target_selected"] == 2  # only the two high-confidence samples

    # Expected Pg is mean softmax over the two confident samples.
    probs = torch.softmax(logits, dim=1)
    expected = probs[:2].mean(dim=0).to(dtype=torch.float64)
    got = torch.as_tensor(captured["target_moments_override"], dtype=torch.float64)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-6, rtol=0)


def test_me_iis_weights_properties(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MEIISConfig(K=1, target_conf_min_count=1, weight_clip_max=2.0, weight_mix_alpha=0.5)
    meiis = MEIIS(featurizer=_DummyFeaturizer(4), feature_dim=4, num_classes=2, seed=0, config=cfg)

    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    target_loader = [_make_target_batch(logits)]
    source_loader = torch.utils.data.DataLoader(_SourceWithIndex(4), batch_size=2, shuffle=False)

    monkeypatch.setattr(meiis, "forward", lambda _x: logits)
    monkeypatch.setattr(meiis.adapter, "fit_target_structure", lambda *args, **kwargs: None)
    monkeypatch.setattr(meiis.adapter, "get_joint_features", lambda _layer_feats, class_probs: {meiis.layer_name: torch.as_tensor(class_probs, dtype=torch.float32).unsqueeze(1)})

    def _fake_solve_iis_from_joint(*args, **kwargs):  # noqa: ANN001
        # Intentionally pathological: negative + huge values.
        return torch.tensor([-1.0, 100.0, 0.0, 0.0], dtype=torch.float64), 0.0, []

    monkeypatch.setattr(meiis.adapter, "solve_iis_from_joint", _fake_solve_iis_from_joint)

    out = meiis.update_importance_weights(source_loader=source_loader, target_loader=target_loader, device=torch.device("cpu"))
    assert out["status"] == "updated"
    w = meiis.source_weights
    assert torch.all(torch.isfinite(w))
    assert torch.all(w >= 0)
    assert float(w.sum().item()) == pytest.approx(1.0, abs=1e-8)


def test_no_target_labels_used_in_me_iis(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = MEIISConfig(K=1, target_conf_min_count=1)
    meiis = MEIIS(featurizer=_DummyFeaturizer(4), feature_dim=4, num_classes=2, seed=0, config=cfg)

    logits = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    source_loader = torch.utils.data.DataLoader(_SourceWithIndex(4), batch_size=2, shuffle=False)

    class _Sentinel:
        def __getattr__(self, name: str):  # noqa: ANN001
            raise RuntimeError(f"target label accessed: {name}")

    target_loader = [_make_target_batch(logits, y=_Sentinel())]

    monkeypatch.setattr(meiis, "forward", lambda _x: logits)
    monkeypatch.setattr(meiis.adapter, "fit_target_structure", lambda *args, **kwargs: None)
    monkeypatch.setattr(meiis.adapter, "get_joint_features", lambda _layer_feats, class_probs: {meiis.layer_name: torch.as_tensor(class_probs, dtype=torch.float32).unsqueeze(1)})
    monkeypatch.setattr(meiis.adapter, "solve_iis_from_joint", lambda *args, **kwargs: (torch.full((4,), 0.25, dtype=torch.float64), 0.0, []))

    out = meiis.update_importance_weights(source_loader=source_loader, target_loader=target_loader, device=torch.device("cpu"))
    assert out["status"] == "updated"

