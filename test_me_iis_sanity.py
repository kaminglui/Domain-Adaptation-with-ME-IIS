import numpy as np
import torch

from models.me_iis_adapter import MaxEntAdapter


def build_toy_joint(num_samples: int, frac_group_a: float, layers) -> torch.Tensor:
    """Construct deterministic joint features for (layer, component, class)."""
    num_group_a = int(num_samples * frac_group_a)
    num_group_b = num_samples - num_group_a
    class_probs = torch.zeros(num_samples, 2, dtype=torch.float32)
    class_probs[:num_group_a, 0] = 1.0
    class_probs[num_group_a:, 1] = 1.0
    joint = {}
    for layer in layers:
        gamma = torch.zeros(num_samples, 2, dtype=torch.float32)
        gamma[:num_group_a, 0] = 1.0
        gamma[num_group_a:, 1] = 1.0
        joint[layer] = gamma.unsqueeze(2) * class_probs.unsqueeze(1)
    assert num_group_a + num_group_b == num_samples
    return joint, class_probs


def run_sanity() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    N_SRC = 200
    N_TGT = 200
    layers = ["layer0", "layer1"]
    components_per_layer = {layer: 2 for layer in layers}
    adapter = MaxEntAdapter(num_classes=2, layers=layers, components_per_layer=components_per_layer, device=device)

    source_joint, source_class_probs = build_toy_joint(N_SRC, frac_group_a=0.5, layers=layers)
    target_joint, target_class_probs = build_toy_joint(N_TGT, frac_group_a=0.75, layers=layers)

    source_flat = adapter.indexer.flatten(source_joint).to(device)
    target_moments = adapter.indexer.flatten(target_joint).to(device).mean(dim=0)

    init_weights = torch.ones(N_SRC, device=device) / float(N_SRC)
    init_moments = torch.sum(init_weights.view(-1, 1) * source_flat, dim=0)
    init_error = float((init_moments - target_moments).abs().max().item())

    weights, _ = adapter.solve_iis(
        source_layer_feats={},
        source_class_probs=source_class_probs.to(device),
        target_layer_feats={},
        target_class_probs=target_class_probs.to(device),
        precomputed_source_joint=source_joint,
        precomputed_target_joint=target_joint,
        max_iter=20,
        iis_tol=0.0,
    )
    final_moments = torch.sum(weights.view(-1, 1) * source_flat, dim=0)
    final_error = float((final_moments - target_moments).abs().max().item())

    print("Synthetic ME-IIS sanity test:")
    print(f"  initial max_abs_error = {init_error:.6f}")
    print(f"  final   max_abs_error = {final_error:.6f}")
    if final_error < 0.1 * init_error:
        print("PASS: final error < 0.1 * initial error")
    else:
        print("FAIL: final error >= 0.1 * initial error")


if __name__ == "__main__":
    run_sanity()

# Quick sanity tests (no dataset needed):
#
#   python test_me_iis_sanity.py
#
# If Office-Home is available at D:\datasets\OfficeHome (for example):
#
#   python train_source.py --data_root D:\datasets\OfficeHome --source_domain Ar --target_domain Cl --num_epochs 1 --batch_size 8 --dry_run_max_batches 5
#
#   python adapt_me_iis.py  --data_root D:\datasets\OfficeHome --source_domain Ar --target_domain Cl --checkpoint checkpoints/source_only_Ar_to_Cl_seed0.pth --dry_run_max_batches 5
