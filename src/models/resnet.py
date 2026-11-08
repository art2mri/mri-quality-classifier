"""MONAI resnet18 backbone with optional MedicalNet-pretrained weights."""

from typing import Literal

import torch
import torch.nn as nn
from monai.networks.nets import resnet18


class ResNetClassifier(nn.Module):
    """
    Wraps a MONAI resnet backbone with a linear classification head, as
    MedicalNet backbone must be built with `feed_forward=False`.
    """

    def __init__(self, backbone: nn.Module, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(in_features=in_features, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)

        return self.fc(features)


def build_resnet18(
    num_classes: int = 2,
    in_channels: int = 1,
    spatial_dims: Literal[2, 3] = 3,
    pretrained: bool = False
) -> nn.Module:
    """
    Build a ResNet18 for MRI classifiation, optionally initialized with
    MedicalNet weights (only available for `spatial_dims=3`).

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels in the MRI image.
        spatial_dims: Number of spatial dimenions used by the model. `2` is
            used for 2D images and `3` for 3D images.
        pretrained: Whether to initialize the backbone with MedicalNet
            pretrained weights.
    """
    if pretrained and spatial_dims == 3:
        backbone = resnet18(
            spatial_dims=spatial_dims,
            n_input_channels=in_channels,
            pretrained=True,
            feed_forward=False,
            shortcut_type='A',
            bias_downsample=True
        )

        return ResNetClassifier(
            backbone=backbone,
            in_features=512,
            num_classes=num_classes
        )

    return resnet18(
        spatial_dims=spatial_dims,
        n_input_channels=in_channels,
        num_classes=num_classes
    )
