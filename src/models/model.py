"""
3D MRI classification model factory using MONAI backbones.
"""

from torch import nn
from monai.networks.nets import resnet18, DenseNet121, EfficientNetBN


def get_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """Returns a MONAI 3D classification model by name."""

    models = {
        'densenet121': lambda: DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes
        ),
        'resnet18': lambda: resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=num_classes
        ),
        'efficientnet-b0': lambda: EfficientNetBN(
            model_name='efficientnet-b0',
            spatial_dims=3,
            in_channels=1,
            num_classes=num_classes,
            pretrained=False
        )
    }

    if model_name not in models:
        raise ValueError(f"Invalid '{model_name}' model.")

    return models[model_name]()
