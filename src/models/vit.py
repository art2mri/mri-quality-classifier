# type: ignore
"""MONAI Vision Transformer wrapped to return only classification logits."""

import torch
import torch.nn as nn
from monai.networks.nets import ViT


class ViTClassifier(nn.Module):
    """
    MONAI viT wrapper, modified to return only the `logits`, as the original
    returns `(logits, hidden_states)`.
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
    img_size: tuple[int, ...] = (256, 256, 256),
    patch_size: tuple[int, ...] = (32, 32, 32),
    spatial_dims: int = 3
) -> nn.Module:
    """
    Compact Vision Transformer for MRI quality classification. `img_size` must
    match the `spatial_size` of the transforms and be divisible by
    `patch_size`.
    """

    return ViTClassifier(
        in_channels=in_channels,
        img_size=img_size,
        patch_size=patch_size,
        hidden_state=384,
        mlp_dim=1536,
        num_layers=6,
        num_heads=6,
        num_classes=num_classes,
        spatial_dims=spatial_dims,
        dropout_rate=0.1
    )
