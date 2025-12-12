import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch.testing import assert_allclose

from clustering.vmf_softmax_backend import VMFSoftmaxBackend, VMFSoftmaxConfig
from models.iis_components import IISUpdater, JointConstraintBuilder
from utils.normalization import l2_normalize


class TestLatentBackendAndConstraints(unittest.TestCase):
    def test_vmf_softmax_pmf_sums_to_one(self) -> None:
        X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        cfg = VMFSoftmaxConfig(kappa=2.5, random_state=0, kmeans_n_init=1, eps=1e-12)
        backend = VMFSoftmaxBackend(n_components=2, config=cfg)
        backend.fit(X)
        probs = backend.predict_proba(X)

        # Row sums must be 1 within numerical tolerance.
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(probs.shape[0]), rtol=0, atol=1e-12)

        # Manual softmax over cosine similarities (explicit loops for clarity).
        X_norm = l2_normalize(X, axis=1, eps=cfg.eps)
        manual = np.zeros_like(probs)
        for i in range(X_norm.shape[0]):
            sims = []
            for k in range(backend.centroids_.shape[0]):
                sims.append(np.dot(X_norm[i], backend.centroids_[k]))
            logits = cfg.kappa * np.array(sims, dtype=np.float64)
            logits -= np.max(logits)
            exp_logits = np.exp(logits)
            manual[i] = exp_logits / exp_logits.sum()
        np.testing.assert_allclose(probs, manual, rtol=0, atol=1e-12)

    def test_joint_constraint_mass_is_Nc(self) -> None:
        device = torch.device("cpu")
        layers = ["l0", "l1"]
        comps = {"l0": 2, "l1": 2}
        class_probs = torch.tensor([[0.2, 0.8], [0.5, 0.5]], dtype=torch.float64, device=device)
        resp = {
            "l0": torch.tensor([[0.6, 0.4], [0.1, 0.9]], dtype=torch.float64, device=device),
            "l1": torch.tensor([[0.3, 0.7], [0.25, 0.75]], dtype=torch.float64, device=device),
        }
        builder = JointConstraintBuilder(layers=layers, components_per_layer=comps, num_classes=2, device=device)
        joint = builder.build_joint_from_responsibilities(resp, class_probs)
        flat, feature_mass = builder.validate_and_flatten(joint, rel_mass_tol=1e-12)

        # For each sample and latent i, sum_{j,c} joint_{i,j,c}(t) = 1, hence sum_{i,j,c} = N_c.
        with torch.no_grad():
            per_layer_mass = torch.stack([joint["l0"].sum(dim=(1, 2)), joint["l1"].sum(dim=(1, 2))], dim=1)
            assert_allclose(per_layer_mass, torch.ones_like(per_layer_mass), rtol=0, atol=1e-12)
            expected_total = torch.full((class_probs.shape[0],), float(len(layers)), dtype=torch.float64)
            assert_allclose(feature_mass.cpu(), expected_total, rtol=0, atol=1e-12)
            expected_total_mass = torch.full((class_probs.shape[0],), float(len(layers)), dtype=torch.float64)
            assert_allclose(flat.sum(dim=1).cpu(), expected_total_mass, rtol=0, atol=1e-12)

    def test_pg_latent_matches_eq14(self) -> None:
        device = torch.device("cpu")
        layers = ["layer"]
        comps = {"layer": 2}
        labels = torch.tensor([0, 1, 0], dtype=torch.long, device=device)
        class_probs = F.one_hot(labels, num_classes=2).to(dtype=torch.float64)
        resp = {"layer": torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.3, 0.7]], dtype=torch.float64, device=device)}
        builder = JointConstraintBuilder(layers=layers, components_per_layer=comps, num_classes=2, device=device)
        joint = builder.build_joint_from_responsibilities(resp, class_probs)
        flat, _ = builder.validate_and_flatten(joint, rel_mass_tol=1e-12)
        iis = IISUpdater(num_latent=len(layers), num_discrete=0, eps=1e-12)
        pg_repo = iis.compute_pg(flat)

        manual = np.zeros((comps["layer"], 2), dtype=np.float64)
        T = labels.shape[0]
        for t in range(T):
            for j in range(comps["layer"]):
                for c in range(2):
                    manual[j, c] += (1 if labels[t].item() == c else 0) * resp["layer"][t, j].item()
        manual /= float(T)
        np.testing.assert_allclose(pg_repo.cpu().numpy(), manual.reshape(-1), rtol=0, atol=1e-12)

    def test_pm_latent_matches_eq15_under_source_hard_labels(self) -> None:
        device = torch.device("cpu")
        layers = ["layer"]
        comps = {"layer": 2}
        labels = torch.tensor([1, 0, 1], dtype=torch.long, device=device)
        class_probs = F.one_hot(labels, num_classes=2).to(dtype=torch.float64)
        resp = {"layer": torch.tensor([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]], dtype=torch.float64, device=device)}
        builder = JointConstraintBuilder(layers=layers, components_per_layer=comps, num_classes=2, device=device)
        joint = builder.build_joint_from_responsibilities(resp, class_probs)
        flat, _ = builder.validate_and_flatten(joint, rel_mass_tol=1e-12)
        weights = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64, device=device)
        iis = IISUpdater(num_latent=len(layers), num_discrete=0, eps=1e-12)
        pm_repo = iis.compute_pm(weights, flat)

        manual = np.zeros((comps["layer"], 2), dtype=np.float64)
        for t in range(labels.shape[0]):
            for j in range(comps["layer"]):
                for c in range(2):
                    manual[j, c] += weights[t].item() * resp["layer"][t, j].item() * (1 if labels[t].item() == c else 0)
        np.testing.assert_allclose(pm_repo.cpu().numpy(), manual.reshape(-1), rtol=0, atol=1e-12)

    def test_iis_delta_lambda_matches_eq18(self) -> None:
        device = torch.device("cpu")
        pg = torch.tensor([0.3, 0.2, 0.1, 0.4], dtype=torch.float64, device=device)
        pm = torch.tensor([0.25, 0.25, 0.2, 0.3], dtype=torch.float64, device=device)
        iis = IISUpdater(num_latent=2, num_discrete=0, eps=1e-12)
        delta = iis.delta_lambda(pg, pm)
        expected = torch.log(torch.clamp(pg, min=1e-12) / torch.clamp(pm, min=1e-12)) / float(iis.mass_constant)
        assert_allclose(delta.cpu(), expected.cpu(), rtol=0, atol=1e-12)

    def test_single_iis_step_updates_weights_correctly(self) -> None:
        device = torch.device("cpu")
        layers = ["l0", "l1"]
        comps = {"l0": 2, "l1": 2}
        builder = JointConstraintBuilder(layers=layers, components_per_layer=comps, num_classes=1, device=device)

        class_probs = torch.ones((2, 1), dtype=torch.float64, device=device)
        source_resp = {
            "l0": torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float64, device=device),
            "l1": torch.tensor([[0.7, 0.3], [0.25, 0.75]], dtype=torch.float64, device=device),
        }
        target_resp = {
            "l0": torch.tensor([[0.5, 0.5], [0.3, 0.7]], dtype=torch.float64, device=device),
            "l1": torch.tensor([[0.6, 0.4], [0.4, 0.6]], dtype=torch.float64, device=device),
        }
        source_joint = builder.build_joint_from_responsibilities(source_resp, class_probs)
        target_joint = builder.build_joint_from_responsibilities(target_resp, class_probs)
        source_flat, _ = builder.validate_and_flatten(source_joint, rel_mass_tol=1e-12)
        target_flat, _ = builder.validate_and_flatten(target_joint, rel_mass_tol=1e-12)

        iis = IISUpdater(num_latent=len(layers), num_discrete=0, eps=1e-12)
        pg = iis.compute_pg(target_flat)
        weights = torch.full((source_flat.shape[0],), 1.0 / source_flat.shape[0], dtype=torch.float64, device=device)
        pm = iis.compute_pm(weights, source_flat)
        delta_expected = torch.log(pg / pm) / float(iis.mass_constant)

        step_result = iis.step(weights, source_flat, pg)
        weights_manual = weights * torch.exp(torch.sum(source_flat * delta_expected.unsqueeze(0), dim=1))
        weights_manual = torch.clamp(weights_manual, min=0.0)
        weights_manual = weights_manual / weights_manual.sum()
        assert_allclose(step_result.delta, delta_expected, rtol=0, atol=1e-12)
        assert_allclose(step_result.updated_weights, weights_manual, rtol=0, atol=1e-12)

        pm_updated = iis.compute_pm(step_result.updated_weights, source_flat)
        l1_before = torch.norm(pg - pm, p=1)
        l1_after = torch.norm(pg - pm_updated, p=1)
        self.assertLessEqual(float(l1_after), float(l1_before) + 1e-12)


if __name__ == "__main__":
    unittest.main()
