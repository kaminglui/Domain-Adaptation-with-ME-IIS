from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Algorithm, unpack_wilds_batch


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GradReverse.apply(x, float(lambd))


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(int(in_dim), int(hidden_dim)), nn.ReLU(inplace=True)]
        if float(dropout) > 0:
            layers.append(nn.Dropout(p=float(dropout)))
        layers.append(nn.Linear(int(hidden_dim), 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DANN(Algorithm):
    def __init__(
        self,
        *,
        featurizer: nn.Module,
        feature_dim: int,
        num_classes: int,
        dann_penalty_weight: float = 1.0,
        grl_lambda: float = 1.0,
        discriminator_hidden_dim: int = 256,
        discriminator_dropout: float = 0.0,
    ):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = nn.Linear(int(feature_dim), int(num_classes))
        self.discriminator = DomainDiscriminator(
            int(feature_dim), hidden_dim=int(discriminator_hidden_dim), dropout=float(discriminator_dropout)
        )
        self.dann_penalty_weight = float(dann_penalty_weight)
        self.grl_lambda = float(grl_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.featurizer(x)
        return self.classifier(feats)

    def update(self, labeled_batch: Any, unlabeled_batch: Any | None = None) -> Dict[str, Any]:
        if unlabeled_batch is None:
            raise ValueError("DANN.update requires an unlabeled_batch.")

        src = unpack_wilds_batch(labeled_batch)
        tgt = unpack_wilds_batch(unlabeled_batch)
        if src.y is None:
            raise ValueError("DANN.update expected labels in labeled_batch but src.y is None.")

        x_s = src.x
        y_s = src.y
        x_t = tgt.x

        x_all = torch.cat([x_s, x_t], dim=0)
        feats_all = self.featurizer(x_all)
        feats_s = feats_all[: x_s.shape[0]]

        logits_s = self.classifier(feats_s)
        loss_cls = F.cross_entropy(logits_s, y_s)
        acc = (logits_s.argmax(dim=1) == y_s).float().mean()

        domain_logits = self.discriminator(grad_reverse(feats_all, self.grl_lambda))
        domain_labels = torch.cat(
            [
                torch.zeros(x_s.shape[0], dtype=torch.long, device=x_all.device),
                torch.ones(x_t.shape[0], dtype=torch.long, device=x_all.device),
            ],
            dim=0,
        )
        loss_domain = F.cross_entropy(domain_logits, domain_labels)
        domain_acc = (domain_logits.argmax(dim=1) == domain_labels).float().mean()

        loss = loss_cls + self.dann_penalty_weight * loss_domain

        return {
            "loss": loss,
            "loss_cls": loss_cls.detach(),
            "loss_domain": loss_domain.detach(),
            "acc": acc.detach(),
            "domain_acc": domain_acc.detach(),
            "batch_size_src": int(x_s.shape[0]),
            "batch_size_tgt": int(x_t.shape[0]),
        }

    def parameter_groups(self) -> Iterable[Dict[str, Any]]:
        return [
            {"name": "featurizer", "params": self.featurizer.parameters()},
            {"name": "classifier", "params": self.classifier.parameters()},
            {"name": "discriminator", "params": self.discriminator.parameters()},
        ]

