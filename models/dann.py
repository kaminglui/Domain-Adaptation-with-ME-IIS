from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def forward(self, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return _GradReverseFn.apply(x, float(lambda_))


@dataclass(frozen=True)
class DomainDiscriminatorConfig:
    hidden_dim: int = 1024
    dropout: float = 0.0


class DomainDiscriminator(nn.Module):
    """MLP domain classifier head used for DANN."""

    def __init__(self, in_dim: int, config: DomainDiscriminatorConfig = DomainDiscriminatorConfig()):
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        dropout = float(config.dropout)
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

