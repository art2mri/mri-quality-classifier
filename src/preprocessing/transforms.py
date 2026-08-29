# type: ignore
"""Preprocessing MONAI transformations for MRI data."""

from typing import Any
import monai.transforms as mt


class EnsureSingleChanneld(mt.MapTransform):
    """
    Selects a single channel from a channel-first image.

    This transform assumes the input image has shape (C, H, W, D) and keeps
    only the specified channel, preserving the channel dimension.
    """

    def __init__(self, keys: list[str], channel_idx: int = 0) -> None:
        super().__init__(keys=keys)
        self.channel_idx = channel_idx

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)

        for key in self.keys:
            data[key] = data[key][self.channel_idx:self.channel_idx+1]

        return data


def get_transforms(preset: str = 'baseline') -> mt.Compose:
    """Returns the transforms pipeline."""

    base_transforms = [
        mt.LoadImaged(keys=['image']),
        mt.EnsureChannelFirstd(keys=['image']),
        EnsureSingleChanneld(keys=['image'], channel_idx=0),
        mt.Orientationd(keys=['image'], axcodes='RAS'),
        mt.Spacingd(
            keys=['image'],
            pixdim=(1.0, 1.0, 1.0),  # Output voxel spacing.
            mode=['bilinear']
        ),
        mt.ResizeWithPadOrCropd(keys=['image'], spatial_size=(256, 256, 256)),
        mt.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)
    ]

    # New transformations, such as augmentation, will come here.

    base_transforms.append(
        mt.EnsureTyped(keys=['image'])  # Input should be a PyTorch tensor.
    )

    return mt.Compose(transforms=base_transforms)
