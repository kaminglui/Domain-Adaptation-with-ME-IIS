import numpy as np
import pytest
import torch
from torch.testing import assert_allclose

from clustering.vmf_softmax_backend import VMFSoftmaxBackend, VMFSoftmaxConfig
from models.me_iis_adapter import MaxEntAdapter
from utils.entropy import prediction_entropy
from utils.normalization import l2_normalize


def _one_layer_joint_from_components(components: torch.Tensor) -> dict:
    return {"layer": components.unsqueeze(2)}


def test_l2_normalize_unit_length() -> None:
    X = np.array([[3.0, 4.0], [0.0, 0.0], [-1.0, 2.0]])
    normed = l2_normalize(X, axis=1, eps=1e-12)
    norms = np.linalg.norm(normed, axis=1)
    assert_allclose(norms[[0, 2]], np.ones(2), rtol=0, atol=1e-12)
    assert np.allclose(normed[1], np.zeros_like(normed[1]))


def test_entropy_matches_definition() -> None:
    probs = np.array([[0.1, 0.6, 0.3], [0.5, 0.25, 0.25]], dtype=np.float64)
    eps = 1e-12
    manual = -np.sum(probs * np.log(probs + eps), axis=1)
    computed = prediction_entropy(probs, eps=eps)
    np.testing.assert_allclose(computed, manual, rtol=0, atol=1e-12)


def test_vmf_softmax_matches_manual_softmax() -> None:
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cfg = VMFSoftmaxConfig(kappa=2.0, random_state=0, kmeans_n_init=1, eps=1e-12)
    backend = VMFSoftmaxBackend(n_components=2, config=cfg)
    backend.fit(X)
    probs = backend.predict_proba(X)

    X_norm = l2_normalize(X, axis=1, eps=cfg.eps)
    sims = X_norm @ backend.centroids_.T
    logits = cfg.kappa * sims
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    manual = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, manual, rtol=0, atol=1e-12)


def test_vmf_softmax_rows_sum_to_one_and_nonnegative() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 3))
    cfg = VMFSoftmaxConfig(kappa=5.0, random_state=1, kmeans_n_init=4, eps=1e-12)
    backend = VMFSoftmaxBackend(n_components=3, config=cfg)
    backend.fit(X)
    probs = backend.predict_proba(X)
    assert probs.min() >= -1e-15
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(probs.shape[0]), rtol=0, atol=1e-12)


def test_joint_features_gamma_times_class_probs() -> None:
    gamma = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float64)
    class_probs = torch.tensor([[0.1, 0.9, 0.0], [0.3, 0.3, 0.4]], dtype=torch.float64)
    joint = gamma.unsqueeze(2) * class_probs.unsqueeze(1)
    manual = torch.zeros_like(joint)
    for n in range(gamma.size(0)):
        for j in range(gamma.size(1)):
            for k in range(class_probs.size(1)):
                manual[n, j, k] = gamma[n, j] * class_probs[n, k]
    assert torch.allclose(joint, manual, atol=1e-12)


def test_target_moments_are_mean_of_joint_features() -> None:
    adapter = MaxEntAdapter(
        num_classes=2, layers=["layer"], components_per_layer={"layer": 2}, device=torch.device("cpu")
    )
    joint_tensor = torch.tensor(
        [
            [[0.6, 0.4], [0.2, 0.8]],
            [[0.3, 0.7], [0.5, 0.5]],
        ],
        dtype=torch.float64,
    )
    joint = {"layer": joint_tensor}
    flat, _ = adapter._validate_joint_features(joint, name="joint", rel_mass_tol=1e-10)
    expected = joint_tensor.reshape(joint_tensor.shape[0], -1).mean(dim=0)
    assert torch.allclose(flat.mean(dim=0), expected, atol=1e-12)


def test_iis_single_step_matches_manual() -> None:
    device = torch.device("cpu")
    adapter = MaxEntAdapter(
        num_classes=1, layers=["layer"], components_per_layer={"layer": 2}, device=device, seed=0
    )
    source_components = torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float64)
    target_components = torch.tensor([[0.7, 0.3], [0.5, 0.5]], dtype=torch.float64)
    source_joint = _one_layer_joint_from_components(source_components)
    target_joint = _one_layer_joint_from_components(target_components)

    weights, _, _ = adapter.solve_iis_from_joint(
        source_joint=source_joint,
        target_joint=target_joint,
        max_iter=1,
        iis_tol=0.0,
        f_mass_rel_tol=1e-12,
    )

    source_flat, feature_mass = adapter._validate_joint_features(
        source_joint, name="source_joint", rel_mass_tol=1e-12
    )
    target_flat, _ = adapter._validate_joint_features(target_joint, name="target_joint", rel_mass_tol=1e-12)
    target_moments = target_flat.to(device).mean(dim=0)
    eps = 1e-8
    weights_manual = torch.ones(source_flat.size(0), dtype=source_flat.dtype) / float(source_flat.size(0))
    mass_constant = float(feature_mass.mean().item())
    current_moments = torch.sum(weights_manual.view(-1, 1) * source_flat, dim=0)
    ratio = torch.ones_like(current_moments)
    active_mask = (target_moments > eps) | (current_moments > eps)
    ratio[active_mask] = (target_moments[active_mask] + eps) / (current_moments[active_mask] + eps)
    ratio = torch.clamp(ratio, min=1e-6, max=1e6)
    delta_update = torch.zeros_like(current_moments)
    delta_update[active_mask] = torch.log(ratio[active_mask]) / (mass_constant + eps)
    exponent = torch.sum(source_flat * delta_update.unsqueeze(0), dim=1)
    expected_weights = weights_manual * torch.exp(exponent)
    expected_weights = torch.clamp(expected_weights, min=0.0)
    expected_weights = expected_weights / expected_weights.sum()

    assert_allclose(weights.cpu(), expected_weights.cpu(), rtol=0, atol=1e-12)


def test_iis_weights_are_probability_distribution() -> None:
    adapter = MaxEntAdapter(
        num_classes=1, layers=["layer"], components_per_layer={"layer": 2}, device=torch.device("cpu"), seed=0
    )
    source_components = torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float64)
    target_components = torch.tensor([[0.9, 0.1], [0.4, 0.6]], dtype=torch.float64)
    source_joint = _one_layer_joint_from_components(source_components)
    target_joint = _one_layer_joint_from_components(target_components)

    weights, _, history = adapter.solve_iis_from_joint(
        source_joint=source_joint,
        target_joint=target_joint,
        max_iter=4,
        iis_tol=0.0,
        f_mass_rel_tol=1e-12,
    )
    assert torch.isfinite(weights).all()
    assert float(weights.sum().item()) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert float(weights.min().item()) >= 0.0
    assert history[-1].max_moment_error >= 0.0


def test_feature_mass_constant_for_valid_gamma_and_class_probs() -> None:
    layers = ["a", "b"]
    comps = {layer: 2 for layer in layers}
    adapter = MaxEntAdapter(num_classes=3, layers=layers, components_per_layer=comps, device=torch.device("cpu"))
    gamma = torch.tensor([[0.6, 0.4], [0.5, 0.5], [0.2, 0.8]], dtype=torch.float64)
    class_probs = torch.tensor(
        [[0.1, 0.7, 0.2], [0.2, 0.3, 0.5], [0.3, 0.1, 0.6]],
        dtype=torch.float64,
    )
    joint = {layer: gamma.unsqueeze(2) * class_probs.unsqueeze(1) for layer in layers}
    _, feature_mass = adapter._validate_joint_features(joint, name="joint", rel_mass_tol=1e-10)
    expected_mass = torch.full((gamma.size(0),), float(len(layers)), dtype=feature_mass.dtype)
    assert torch.allclose(feature_mass, expected_mass, atol=1e-12)


def test_adapter_runs_with_vmf_backend_small_synthetic() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    adapter = MaxEntAdapter(
        num_classes=2,
        layers=["layer"],
        components_per_layer={"layer": 2},
        device=device,
        seed=0,
        cluster_backend="vmf_softmax",
        vmf_kappa=15.0,
        kmeans_n_init=4,
    )

    source_feats = {"layer": torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=torch.float64)}
    target_feats = {"layer": torch.tensor([[1.0, 0.0], [1.0, 0.2], [-1.0, -0.1]], dtype=torch.float64)}
    source_class_probs = torch.tensor(
        [[0.9, 0.1], [0.85, 0.15], [0.2, 0.8], [0.15, 0.85]], dtype=torch.float64
    )
    target_class_probs = torch.tensor([[0.7, 0.3], [0.6, 0.4], [0.1, 0.9]], dtype=torch.float64)

    adapter.fit_target_structure(target_feats, target_class_probs=target_class_probs)

    source_joint = adapter.get_joint_features(source_feats, source_class_probs)
    target_joint = adapter.get_joint_features(target_feats, target_class_probs)
    target_flat, _ = adapter._validate_joint_features(target_joint, name="target_joint", rel_mass_tol=1e-6)
    source_flat, _ = adapter._validate_joint_features(source_joint, name="source_joint", rel_mass_tol=1e-6)
    target_moments = target_flat.mean(dim=0)
    init_weights = torch.ones(source_flat.size(0), dtype=source_flat.dtype) / float(source_flat.size(0))
    init_moments = torch.sum(init_weights.view(-1, 1) * source_flat, dim=0)
    init_error = float((init_moments - target_moments).abs().max().item())

    weights, history = adapter.solve_iis(
        source_layer_feats=source_feats,
        source_class_probs=source_class_probs,
        target_layer_feats=target_feats,
        target_class_probs=target_class_probs,
        max_iter=6,
        iis_tol=0.0,
        f_mass_rel_tol=1e-3,
    )
    final_error = history[-1].max_moment_error if history else float("inf")
    assert float(weights.sum().item()) == pytest.approx(1.0, rel=0, abs=1e-12)
    assert float(weights.min().item()) >= 0.0
    assert final_error < init_error
