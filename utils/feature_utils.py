from typing import Dict, List, Tuple

import torch
from tqdm.auto import tqdm


@torch.no_grad()
def extract_features(
    model,
    loader,
    device: torch.device,
    feature_layers,
    max_batches: int = 0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Run the model over the loader, capturing pooled activations for given feature_layers.

    Returns:
        layer_feats: Dict[layer_name, torch.Tensor]  # stacked features (CPU)
        logits:      torch.Tensor                    # logits on CPU
        labels:      torch.Tensor                    # labels tensor
    """
    model.eval()
    feats: Dict[str, List[torch.Tensor]] = {layer: [] for layer in feature_layers}
    logits_list: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    batch_count = 0
    for batch in tqdm(loader, desc="Extract", leave=False):
        if len(batch) == 2:
            images, batch_labels = batch
        else:
            images, batch_labels = batch[0], batch[1]
        images = images.to(device)
        logits, _, inter = model.forward_with_intermediates(images, feature_layers=feature_layers)
        logits_list.append(logits.cpu())
        labels.append(batch_labels)
        for layer in feature_layers:
            feats[layer].append(inter[layer].cpu())
        batch_count += 1
        if max_batches > 0 and batch_count >= max_batches:
            break

    if not logits_list:
        raise RuntimeError("No batches were processed during feature extraction.")

    layer_feats = {layer: torch.cat(feats[layer]) for layer in feature_layers}
    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels)
    return layer_feats, logits_all, labels_all
