# type: ignore
"""Preprocessing MONAI transformations for 2D and 3D MRI data."""

from typing import Any, Literal
import monai.transforms as mt


class EnsureSingleChanneld(mt.MapTransform):
    """
    Keeps only one channel from a channel-first image. This transform assumes
    the input image has shape (C, H, W, D) and keeps only the specified
    channel, preserving the channel dimension with the shape (1, H, W, D).

    Args:
        keys: Dictionary keys containing the images to transform.
        channel_idx: Shape index to be preserved.
    """

    def __init__(self, keys: list[str], channel_idx: int = 0) -> None:
        super().__init__(keys=keys)
        self.channel_idx = channel_idx

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)

        for key in self.keys:
            data[key] = data[key][self.channel_idx:self.channel_idx + 1]

        return data


class ExtractSagittalRanged(mt.MapTransform):
    """
    Extract multiple sagittal slices from a 3D channel-first image with shape
    (C, X, Y, Z), converting each one to a 2D image with shape (C, Y, Z). It
    assumes `RAS` orientation, so axis 1 (X) always corresponds to the
    left-right axis.

    This is a multi-sample transform: it receives one dict and returns a list
    of dicts, one per offset.

    Args:
        keys: Dictionary keys containing the images to transform.
        offsets: Offsets relative to the central sagittal slice, one output
            sample is generated per offset.
    """

    def __init__(self, keys: list[str], offsets: list[int]) -> None:
        super().__init__(keys=keys)
        self.offsets = offsets

    def __call__(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        data = dict(data)
        samples = []

        for offset in self.offsets:
            sample = dict(data)

            for key in self.keys:
                volume = data[key]  # Shape (C, X, Y, Z)
                size_x = volume.shape[1]

                slice_idx = size_x // 2 + offset
                slice_idx = max(0, min(slice_idx, size_x - 1))

                sample[key] = volume[:, slice_idx, :, :]

            sample['sagittal_offset'] = offset
            samples.append(sample)

        return samples


def get_transforms(
    preset: str = 'baseline',
    mode: Literal['2d', '3d'] = '3d',
    sagittal_offsets: list[int] = [0]
) -> mt.Compose:
    """
    Build the preprocessing pipeline for 2D or 3D training.

    Args:
        preset: Name of the preprocessing preset. Currently, only `baseline` is
            available.
        mode: Defines whether the output is a 2D or 3D image.
        sagittal_offset: Offsets from the central sagittal slice, each one
            generating a separate 2D sample. It is used onlye when
            `mode='2d'`.
    """

    transforms = [
        # Reading images.
        mt.LoadImaged(keys=['image']),

        # Ensuring channel dimension.
        mt.EnsureChannelFirstd(keys=['image']),
        EnsureSingleChanneld(keys=['image'], channel_idx=0),

        # Reorient the volume to a consistent RAS orientation.
        mt.Orientationd(keys=['image'], axcodes='RAS'),

        # Resampling the volume to isotropic 1mm voxel spacing.
        mt.Spacingd(
            keys=['image'],
            pixdim=(1.0, 1.0, 1.0),
            mode='bilinear'
        )
    ]

    if mode == '3d':
        transforms.append(
            mt.ResizeWithPadOrCropd(keys=['image'], spatial_size=(256, 256, 256))
        )
    else:
        transforms.extend([
            ExtractSagittalRanged(keys=['image'], offsets=sagittal_offsets),
            mt.ResizeWithPadOrCropd(keys='image', spatial_size=(256, 256))
        ])

    transforms.extend([
        # Normalizing with z-score.
        mt.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),

        # Ensuring to be a PyTorch tensor.
        mt.EnsureTyped(keys=['image'])
    ])

    return mt.Compose(transforms=transforms)
