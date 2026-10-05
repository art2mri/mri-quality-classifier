"""VGG backbone from torchvision, adapted for single-channel 2D input."""

import torch.nn as nn
from torchvision.models import vgg16


def build_vgg16(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    """
    VGG16 wrapper since the defaul model assumes a 3-channel RGB input, and
    MRI slices are single-channel.
    """
    model = vgg16(weights=None)

    model.features[0] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,
        padding=1
    )
    model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=num_classes
    )

    return model
