"""
Utilities for preparing train/test splits.
"""

from pathlib import Path
from typing import Any


def _to_binary(label: Any, positive_class: str = 'not-useful') -> int:
    if isinstance(label, str):
        return int(label == positive_class)
    return int(label)


def _convert(
    items: list[dict],
    images_dir: Path,
    positive_class: str = 'not-useful'
) -> list[dict]:
    return [
        {
            'image': str(images_dir / item['image']),
            'label': _to_binary(label=item['label'], positive_class=positive_class)  # noqa: E501
        }
        for item in items
    ]


def prepare_split(
    split_data: dict,
    images_dir: str | Path,
    positive_class: str = 'not-useful'
) -> dict[str, list]:
    """
    Prepare train/test split dictionaries for data loading.

    Args:
        split_data: dict containing the split samples. Expected keys are
            'train' and 'test', each mapping to a list of dicts with at least
            'image' and 'label'.
        images_dir: base directory where the image files are stored.
        positive_class: label value that should be converted to 1.
    """
    images_dir = Path(images_dir)

    return {
        'train': _convert(
            items=split_data['train'],
            images_dir=images_dir,
            positive_class=positive_class
        ),
        'test': _convert(
            items=split_data['test'],
            images_dir=images_dir,
            positive_class=positive_class
        )
    }
