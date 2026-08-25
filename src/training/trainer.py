"""Trainer class."""

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class MRIQualityTrainer:
    """TO DO."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        optimizer: Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: str | None = None
    ) -> None:
        pass
