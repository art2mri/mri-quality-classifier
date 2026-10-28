# type: ignore
"""VGG backbone from torchvision, adapted for single-channel 2D input."""

import torch.nn as nn
from torchvision.models import vgg16


def build_vgg16(num_classes: int = 2, in_channels: int = 1) -> nn.Module:
    """
    Build an adapted VGG16 2D for single-channel MRI images (since the default
    model assumes a 3-channel RGB input).

    The only adaptation is to replace the first convolution layer to accept
    `in_channels` channels.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels in the MRI image.
    """
    model = vgg16(weights=None)

    model.features[0] = nn.Conv2d(
        in_channels=in_channels,  # Replacing for single-channel MRI images.
        out_channels=64,
        kernel_size=3,
        padding=1
    )
    model.classifier[6] = nn.Linear(
        in_features=4096,
        out_features=num_classes
    )

    return model
