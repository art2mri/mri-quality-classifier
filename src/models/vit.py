# type: ignore
"""MONAI Vistion Transformer adapted to return only logits."""

from typing import Literal

import torch
import torch.nn as nn
from monai.networks.nets import ViT

VIT_DEFAULTS = {
    2: {'img_size': (256, 256), 'patch_size': (16, 16)},
    3: {'img_size': (256, 256, 256), 'patch_size': (32, 32, 32)}
}


class ViTClassifier(nn.Module):
    """
    MONAI Vision Transformer wrapper, modified to return only the `logits`, as
    the original implementation returns `(logits, hidden_states)`.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.vit = ViT(classification=True, post_activation=None, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.vit(x)  # Drop hidden_states.

        return logits


def build_vit(
    num_classes: int = 2,
    in_channels: int = 1,
    spatial_dims: Literal[2, 3] = 3,
    img_size: tuple[int, ...] | None = None,
    patch_size: tuple[int, ...] | None = None
) -> nn.Module:
    """
    Build an adapted Vision Transformer for MRI quality classification.

    If `img_size` or `patch_size` are not given, defaults for `spatial_dims`
    are used. The `img_size` must match the `spatial_size` of the transforms
    and be divisible by `patch_size`.

    Args:
        num_classes: Number of output classes.
        in_channels: Number of input channels in the MRI image.
        spatial_dims: Number of spatial dimensions used by the model. `2` is
            used for 2D images and `3` for 3D images.
        img_size: Dimension of input image.
        patch_size: Patch length.
    """
    defaults = VIT_DEFAULTS[spatial_dims]

    return ViTClassifier(
        in_channels=in_channels,
        img_size=img_size or defaults['img_size'],
        patch_size=patch_size or defaults['patch_size'],
        hidden_size=384,  # We can use 192 for fastter tests.
        mlp_dim=1536,  # The literature consensus is to use 4 * hidden_size.
        num_layers=6,
        num_heads=6,  # Should use 6 for hidden_size=384.
        num_classes=num_classes,
        spatial_dims=spatial_dims,
        dropout_rate=0.1
    )
