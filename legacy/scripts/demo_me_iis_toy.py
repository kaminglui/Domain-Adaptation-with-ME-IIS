"""
Tiny, dataset-free ME-IIS demo on a synthetic joint distribution.
"""
import torch

from models.me_iis_adapter import MaxEntAdapter


def _make_joint(components, class_probs, layer_name: str, num_components: int, num_classes: int):
    joint = torch.zeros(len(components), num_components, num_classes, dtype=torch.float32)
    for idx, comp in enumerate(components):
        joint[idx, comp, :] = class_probs[idx]
    return {layer_name: joint}


def build_toy():
    """
    Construct a toy source/target joint along with class probabilities.
    Source leans toward class 0 / component 0; target leans toward class 1 / component 1.
    """
    num_components = 2
    num_classes = 2
    layer_name = "layer"

    source_components = [0, 0, 1, 1]
    source_class_probs = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]], dtype=torch.float32
    )
    target_components = [0, 1, 1, 1]
    target_class_probs = torch.tensor(
        [[1.0, 0.0], [0.2, 0.8], [0.1, 0.9], [0.0, 1.0]], dtype=torch.float32
    )

    source_joint = _make_joint(source_components, source_class_probs, layer_name, num_components, num_classes)
    target_joint = _make_joint(target_components, target_class_probs, layer_name, num_components, num_classes)
    return source_joint, target_joint, source_class_probs, target_class_probs


def main():
    torch.manual_seed(0)
    device = torch.device("cpu")
    source_joint, target_joint, source_class_probs, target_class_probs = build_toy()
    adapter = MaxEntAdapter(
        num_classes=2, layers=["layer"], components_per_layer={"layer": 2}, device=device, seed=0
    )
    weights, final_error, history = adapter.solve_iis_from_joint(
        source_joint=source_joint,
        target_joint=target_joint,
        max_iter=10,
        iis_tol=0.0,
        f_mass_rel_tol=1e-6,
    )

    p_target = target_class_probs.mean(dim=0)
    p_source = source_class_probs.mean(dim=0)
    p_weighted = (weights.unsqueeze(1) * source_class_probs).sum(dim=0)

    print("[IIS Demo] Class marginals comparison:")
    for cls_idx in range(p_target.numel()):
        print(
            f"  Class {cls_idx}: "
            f"P_target={float(p_target[cls_idx]):.3f}, "
            f"P_weighted_source={float(p_weighted[cls_idx]):.3f}, "
            f"P_source={float(p_source[cls_idx]):.3f}"
        )
    print(f"[IIS Demo] Final max moment error: {history[-1].max_moment_error:.4e}")
    print(f"[IIS Demo] Sample weights: {[round(float(w), 4) for w in weights.tolist()]}")


if __name__ == "__main__":
    main()
