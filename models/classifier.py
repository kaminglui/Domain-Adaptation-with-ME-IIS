import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.fc.weight, mode="fan_out", nonlinearity="relu")
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_model(
    num_classes: int,
    pretrained: bool = True,
    *,
    bottleneck_dim: int = 256,
    bottleneck_bn: bool = True,
    bottleneck_relu: bool = True,
    bottleneck_dropout: float = 0.0,
):
    from models.backbone import ResNet50Backbone
    from models.uda_model import Bottleneck, UdaModel

    backbone = ResNet50Backbone(pretrained=pretrained)
    bottleneck = Bottleneck(
        backbone.out_features,
        int(bottleneck_dim),
        use_bn=bool(bottleneck_bn),
        use_relu=bool(bottleneck_relu),
        dropout=float(bottleneck_dropout),
    )
    head = ClassifierHead(bottleneck.out_features, num_classes)
    return UdaModel(backbone, bottleneck, head)
