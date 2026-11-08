# type: ignore
"""Factory for 2D and 3D MRI classification models using MONAI backbones."""

from typing import Literal

import torch.nn as nn
from src.models.vit import build_vit
from src.models.vgg import build_vgg16
from monai.networks.nets import resnet18, DenseNet121, EfficientNetBN

Models = Literal[
    'densenet121', 'resnet18', 'efficientnet-b0', 'vgg16', 'vit'
]

PRETRAINED_2D_MODELS = {'densenet121', 'efficientnet-b0', 'vgg16'}


def get_model(
    model_name: Models,
    num_classes: int = 2,
    spatial_dims: Literal[2, 3] = 3,
    pretrained: bool = False
) -> nn.Module:
    """
    Build a MRI classification model.

    Args:
        model_name: Name of the architecture to use.
        num_classes: Number of output classes. Currently, only `2` is accepted.
        spatial_dims: Number of spatial dimensions used by the model. `2` is
            used for 2D images with shape `(C, H, W)` and `3` is used for 3D
            images with shape `(C, D, H, W)`.
        pretrained: Whether to initialize the backbone with ImageNet-pretrained
            weights.
    """
    if pretrained and model_name in PRETRAINED_2D_MODELS and spatial_dims != 2:
        raise ValueError(f"'{model_name}' only supports pretrained weights when spatial_dims=2.")
    if pretrained and model_name not in PRETRAINED_2D_MODELS:
        raise ValueError(f"'{model_name}' does not support pretrained weights.")

    models = {
        'densenet121': lambda: DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=1,  # MRI image has only 1 channel.
            out_channels=num_classes,
            pretrained=pretrained
        ),
        'resnet18': lambda: resnet18(
            spatial_dims=spatial_dims,
            n_input_channels=1,
            num_classes=num_classes
        ),
        'efficientnet-b0': lambda: EfficientNetBN(
            model_name='efficientnet-b0',
            spatial_dims=spatial_dims,
            in_channels=1,
            num_classes=num_classes,
            pretrained=pretrained
        ),
        'vgg16': lambda: build_vgg16(
            num_classes=num_classes,
            in_channels=1,
            pretrained=pretrained
        ),
        'vit': lambda: build_vit(
            num_classes=num_classes,
            in_channels=1,
            spatial_dims=spatial_dims
        )
    }

    if model_name == 'vgg16' and spatial_dims != 2:
        raise ValueError("'vgg16' only supports spatial_dims=2.")
    if model_name not in models:
        raise ValueError(f"Invalid '{model_name}' model.")

    return models[model_name]()
