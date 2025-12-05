from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> Tuple[float, np.ndarray]:
    """Compute top-1 accuracy and confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, _ = model(images, return_features=False)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    acc = float((y_true == y_pred).mean() * 100.0)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


@torch.no_grad()
def predict_features(model, loader: DataLoader, device: torch.device):
    """Return features, logits, labels for full loader."""
    model.eval()
    feats, logits_list, labels_list = [], [], []
    for batch in loader:
        if len(batch) == 2:
            images, labels = batch
        else:
            images, labels = batch[0], batch[1]
        images = images.to(device)
        labels_list.append(labels)
        logits, features = model(images, return_features=True)
        feats.append(features.cpu())
        logits_list.append(logits.cpu())
    features_all = torch.cat(feats)
    logits_all = torch.cat(logits_list)
    labels_all = torch.cat(labels_list)
    return features_all, logits_all, labels_all
