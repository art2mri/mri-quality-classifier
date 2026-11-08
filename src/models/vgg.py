# type: ignore
"""VGG backbone from torchvision, adapted for single-channel 2D input."""

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


def build_vgg16(
    num_classes: int = 2,
    in_channels: int = 1,
    pretrained: bool = False
) -> nn.Module:
    """
    Build an adapted VGG 2D for single-channel MRI images (since the default
    model assumes a 3-channel RGB input).

    The first conv layer is always replaced to accept `in_channels` channels,
    so when `pretrained=True`, only that first layer keeps its random
    initialization (every other layer reuses the ImageNet weights).

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels in the MRI image.
        pretrained: Whether to load ImageNet-pretrained weights.
    """
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = vgg16(weights=weights)

    model.features[0] = nn.Conv2d(
        in_channels=in_channels,  # Replacing for 1-channel MRI images.
        out_channels=64,
        kernel_size=3,
        padding=1
    )
    model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=num_classes  # 2 is default.
    )

    return model
