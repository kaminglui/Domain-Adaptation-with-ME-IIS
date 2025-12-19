from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch

from models.me_iis_adapter import MaxEntAdapter
from src.datasets.wilds_camelyon17 import build_camelyon17_loaders
from src.train.train_loop import train
from src.utils.run_id import decode_run_id_to_config, encode_config_to_run_id


def test_run_id_encode_decode_stability() -> None:
    cfg1 = {
        "dataset": "camelyon17",
        "split_mode": "uda_target",
        "algorithm": "MEIIS",
        "backbone": "densenet121",
        "seed": 0,
        "lr": 1e-4,
        "batch_size": 64,
        "grad_accum_steps": 2,
        "meiis_K": 8,
        "meiis_tau": 0.9,
    }
    cfg2 = dict(reversed(list(cfg1.items())))  # different order
    rid1 = encode_config_to_run_id(cfg1)
    rid2 = encode_config_to_run_id(cfg2)
    assert rid1 == rid2
    assert " " not in rid1
    assert "/" not in rid1 and "\\" not in rid1
    dec = decode_run_id_to_config(rid1)
    assert dec.get("dataset") == "camelyon17"
    assert dec.get("algorithm") == "MEIIS"


class _StubSubset(torch.utils.data.Dataset):
    def __init__(self, split: str, n: int = 8):
        self.split = split
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        x = torch.zeros(3, 8, 8) + float(idx)
        y = torch.tensor(0 if "unlabeled" in self.split else idx % 2, dtype=torch.long)
        metadata = torch.tensor([5 if "test" in self.split else 4 if "val" in self.split else 1], dtype=torch.long)
        return x, y, metadata


class _StubWILDSDataset:
    def __init__(self):
        self.subsets: Dict[str, _StubSubset] = {}

    def get_subset(self, split: str, transform=None):  # noqa: ANN001
        ds = _StubSubset(split)
        self.subsets[split] = ds
        return ds

    def eval(self, y_pred, y_true, metadata):  # noqa: ANN001
        # Minimal accuracy metric; accept either logits or predicted labels.
        y_pred_t = torch.as_tensor(y_pred)
        if y_pred_t.ndim == 2 and y_pred_t.shape[1] > 1:
            y_pred_t = y_pred_t.argmax(dim=1)
        y_true_t = torch.as_tensor(y_true)
        acc = float((y_pred_t == y_true_t).float().mean().item())
        return {"acc_avg": acc}


def test_wilds_camelyon17_loader_shapes_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.datasets.wilds_camelyon17 as mod

    stub = _StubWILDSDataset()

    def _fake_get_dataset(*, root_dir: str, download: bool, unlabeled: bool):  # noqa: ANN001
        return stub

    def _fake_get_loaders():  # noqa: ANN001
        def _train_loader(_typ: str, dataset, batch_size: int, **kwargs):  # noqa: ANN001
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        def _eval_loader(_typ: str, dataset, batch_size: int, **kwargs):  # noqa: ANN001
            return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return _train_loader, _eval_loader

    monkeypatch.setattr(mod, "get_camelyon17_dataset", _fake_get_dataset)
    monkeypatch.setattr(mod, "_get_wilds_loaders", _fake_get_loaders)

    loaders = build_camelyon17_loaders(
        {
            "data_root": "unused",
            "download": False,
            "unlabeled": True,
            "split_mode": "align_val",
            "eval_split": "val",
            "adapt_split": "val_unlabeled",
            "train_transform": None,
            "eval_transform": None,
            "batch_size": 4,
            "include_indices_in_train": True,
            "num_workers": 0,
        }
    )

    batch = next(iter(loaders["train_loader"]))
    assert len(batch) == 4  # x, y, metadata, idx
    x, y, metadata, idx = batch
    assert x.shape == (4, 3, 8, 8)
    assert y.shape == (4,)
    assert metadata.shape == (4, 1)
    assert idx.shape == (4,)

    # split_mode=align_val should use val_unlabeled for unlabeled_loader
    assert loaders["unlabeled_loader"] is not None
    assert loaders["unlabeled_loader"].dataset is stub.subsets["val_unlabeled"]


class _TinyWildsDS:
    def eval(self, y_pred, y_true, metadata):  # noqa: ANN001
        y_pred_t = torch.as_tensor(y_pred)
        if y_pred_t.ndim == 2 and y_pred_t.shape[1] > 1:
            y_pred_t = y_pred_t.argmax(dim=1)
        y_true_t = torch.as_tensor(y_true)
        acc = float((y_pred_t == y_true_t).float().mean().item())
        return {"acc_avg": acc, "acc_wg": acc}, "stub"


class _Tiny3Tuple(torch.utils.data.Dataset):
    def __init__(self, n: int = 8):
        self.x = torch.randn(n, 4)
        self.y = torch.randint(0, 2, (n,))
        self.m = torch.zeros(n, 1, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.m[idx]


class _ERMAlgo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(4, 2)

    def predict(self, x):  # noqa: ANN001
        return self.net(x)

    def update(self, batch_raw, unlabeled_batch=None):  # noqa: ANN001
        x, y, _m = batch_raw
        logits = self.net(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return {"loss": loss, "acc": acc, "batch_size": int(x.shape[0])}

    def parameter_groups(self):
        return [{"params": self.parameters(), "name": "all"}]


class _BrokenAlgo(_ERMAlgo):
    def update(self, batch_raw, unlabeled_batch=None):  # noqa: ANN001
        raise RuntimeError("should not be called when skip logic works")


def test_checkpoint_skip_logic(tmp_path: Path) -> None:
    run_dir = tmp_path / "rid"
    ds = _TinyWildsDS()
    train_ds = _Tiny3Tuple(16)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=False)

    cfg = {
        "run_id": "rid",
        "seed": 0,
        "epochs": 1,
        "batch_size": 8,
        "grad_accum_steps": 1,
        "optimizer": "adamw",
        "lr": 1e-3,
        "weight_decay": 0.0,
        "early_stop_patience": 0,
        "amp": False,
        "deterministic": True,
        "device": "cpu",
        "resume": False,
    }

    res1 = train(
        cfg=cfg,
        run_dir=run_dir,
        algorithm=_ERMAlgo(),
        wilds_dataset=ds,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        unlabeled_loader=None,
        id_val_loader=None,
    )
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "results.json").exists()

    res2 = train(
        cfg=cfg,
        run_dir=run_dir,
        algorithm=_BrokenAlgo(),
        wilds_dataset=ds,
        train_loader=loader,
        val_loader=loader,
        test_loader=loader,
        unlabeled_loader=None,
        id_val_loader=None,
    )
    assert res2["status"] == "skipped"


def test_me_iis_objective_monotonic_on_toy_data() -> None:
    adapter = MaxEntAdapter(num_classes=2, layers=["layer"], components_per_layer={"layer": 2}, device=torch.device("cpu"))

    # Source: 4 samples, each activates one (component, class).
    source = torch.zeros((4, 2, 2), dtype=torch.float32)
    source[0, 0, 0] = 1.0
    source[1, 0, 1] = 1.0
    source[2, 1, 0] = 1.0
    source[3, 1, 1] = 1.0

    # Target proportions [0.4,0.1,0.2,0.3].
    target = torch.zeros((10, 2, 2), dtype=torch.float32)
    target[0:4, 0, 0] = 1.0
    target[4:5, 0, 1] = 1.0
    target[5:7, 1, 0] = 1.0
    target[7:10, 1, 1] = 1.0

    _w, _err, history = adapter.solve_iis_from_joint(
        source_joint={"layer": source},
        target_joint={"layer": target},
        max_iter=5,
        iis_tol=0.0,
    )
    obj = [h.objective for h in history]
    assert len(obj) > 1
    for prev, curr in zip(obj, obj[1:]):
        assert curr >= prev - 1e-8


def test_me_iis_weights_nonnegative_and_not_all_zero() -> None:
    adapter = MaxEntAdapter(num_classes=1, layers=["layer"], components_per_layer={"layer": 2}, device=torch.device("cpu"))
    source = torch.tensor([[[1.0], [0.0]], [[0.0], [1.0]]], dtype=torch.float32)
    target = torch.tensor([[[1.0], [0.0]], [[1.0], [0.0]], [[0.0], [1.0]]], dtype=torch.float32)
    w, _err, _hist = adapter.solve_iis_from_joint(source_joint={"layer": source}, target_joint={"layer": target}, max_iter=5)
    assert torch.all(w >= 0)
    assert float(w.sum().item()) == pytest.approx(1.0, abs=1e-8)
    assert torch.any(w > 0)
