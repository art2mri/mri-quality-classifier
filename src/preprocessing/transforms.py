# type: ignore
import monai.transforms as mt


def get_transforms(preset: str = 'baseline') -> mt.Compose:
    """Returnsthe transforms pipeline."""

    base_transforms = [
        mt.LoadImaged(keys=['image']),
        mt.EnsureChannelFirstd(keys=['image']),  # Adjust the channel dimension of input data.
        mt.Orientationd(keys=['image'], axcodes='RAS'),
        mt.Spacingd(
            keys=['image'],
            pixdim=(1.0, 1.0, 1.0),  # Output voxel spacing.
            mode=['bilinear']
        ),
        mt.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True)
    ]

    # New transformations, such as augmentation, will come here.

    base_transforms.append(
        mt.EnsureTyped(keys=['image'])  # Ensure the input data to be a PyTorch tensor.
    )

    return mt.Compose(transforms=base_transforms)
