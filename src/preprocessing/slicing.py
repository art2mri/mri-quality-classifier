"""Slicing functions for MRI preprocessing."""

from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image


def normalize_to_uint8(
    array: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Normalize a 2D array to the 0-255 uint8 range using percentile clipping.
    """
    # Compute the clipping bounds based on percentiles.
    low = np.percentile(array, lower_percentile)
    high = np.percentile(array, upper_percentile)

    # Avoid division by zero if the slice is flat.
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)

    # Clip values outside the range, then scale to [0, 255].
    clipped = np.clip(array, low, high)
    scaled = (clipped - low) / (high - low) * 255.0

    return scaled.astype(np.uint8)


def extract_sagittal_slice(
    input_path: str | Path,
    output_path: str | Path,
    slice_index: int | None = None,
    offset: int = 0
) -> Path:
    """
    Extract a single sagittal slice from a 3D NIfTI volume and save it as a 2D
    image.

    Args:
        input_path: path to the input .nii.gz file.
        output_path: path where the output image will be saved.
        slice_index: index along the sagittal axis to extract. If None, the
            central slice is used.
        offset: offset applied to the central slice when slice_index is None.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    img = nib.load(filename=str(input_path))  # Load image.
    img = nib.as_closest_canonical(img=img)  # Reorient to canonical (RAS+).
    data = img.get_fdata()  # Get the raw voxel as a numpy array.

    if data.ndim == 4:
        data = data[..., 0]

    if slice_index is None:
        slice_index = data.shape[0] // 2 + offset

    slice_2d = data[slice_index, :, :]  # Get the 2D sagittal slice.
    slice_uint8 = normalize_to_uint8(array=slice_2d)  # Normalize float intensities.
    slice_uint8 = np.rot90(m=slice_uint8)  # Rotate for conventional display.

    # Convert to PIL and save.
    image = Image.fromarray(obj=slice_uint8, mode='L')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return output_path
