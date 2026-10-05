# type: ignore
"""
Factory for 2D and 3D MRI classification models using MONAI backbones.
"""

from typing import Literal
import torch.nn as nn
from monai.networks.nets import resnet18, DenseNet121, EfficientNetBN
from src.models.vgg import build_vgg16


def get_model(
    model_name: str,
    num_classes: int = 2,
    spatial_dims: Literal[2, 3] = 3
) -> nn.Module:
    """
    Returns a MONAI classification model.

    Args:
        model_name: Name of the architecture to use.
        num_classes: Number of output classes.
        spatial_dims: Number of spatial dimensions used by the model. 2 is used
            for 2D images with shape `(C, H, W)` and 3 is used for 3D images
            with shape `(C, D, H, W)`.
    """

    models = {
        'densenet121': lambda: DenseNet121(
            spatial_dims=spatial_dims,
            in_channels=1,
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
        )
    }

    if model_name == 'vgg16' and spatial_dims != 2:
        raise ValueError("'vgg16' from torchvision only supports spatial_dims=2.")

    if model_name not in models:
        raise ValueError(f"Invalid '{model_name}' model.")

    return models[model_name]()
