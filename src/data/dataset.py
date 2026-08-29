# type: ignore
"""Utilities for building MONAI datasets and dataloaders."""

from monai.transforms import Compose
from monai.data import DataLoader, Dataset


def make_loaders(
    train_data: list[dict],
    val_data: list[dict],
    transforms: Compose,
    batch_size: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders from sample lists."""

    train_dataset = Dataset(data=train_data, transform=transforms)
    val_dataset = Dataset(data=val_data, transform=transforms)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader
