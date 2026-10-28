# type: ignore
"""
Factory for 2D and 3D MRI classification models using MONAI backbones.
"""

from typing import Literal

import torch.nn as nn
from src.models.vit import build_vit
from src.models.vgg import build_vgg16
from monai.networks.nets import resnet18, DenseNet121, EfficientNetBN

Models = Literal[
    'densenet121', 'resnet18', 'efficientnet-b0', 'vgg16', 'vit'
]


def get_model(
    model_name: Models,
    num_classes: int = 2,
    spatial_dims: Literal[2, 3] = 3
) -> nn.Module:
    """
    Build a MRI classification model.

    All models return raw logits, without the application of softmax. These
    logits can be used directly with `torch.nn.CrossEntropyLoss`.

    Args:
        model_name: Name of the architecture to use.
        num_classes: Number of output classes. Currently, only `2` is accepted.
        spatial_dims: Number of spatial dimensions used by the model. `2` is
            used for 2D images with shape `(C, H, W)` and `3` is used for 3D
            images with shape `(C, D, H, W)`.
    """

    models = {
        'densenet121': lambda: DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=1,  # MRI has only 1 channel.
            out_channels=num_classes
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
            pretrained=False
        ),
        'vgg16': lambda: build_vgg16(
            num_classes=num_classes,
            in_channels=1
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
