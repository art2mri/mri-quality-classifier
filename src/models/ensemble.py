"""Soft-voting ensemble for MRI quality classification models."""

import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    """
    Soft-voting ensemble.
    """

    def __init__(
        self,
        models: list[nn.Module],
        weights: list[float] | None = None,
        threshold: float = 0.5
    ) -> None:
        super().__init__()

        self.models = nn.ModuleList(models)
        self.weights = torch.tensor(weights or [1.0] * len(models))
        self.weights = self.weights / self.weights.sum()
        self.threshold = threshold

        for model in self.models:
            model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Custom forward pass returning log-probabilities, as inference module
        needs to calculate the softmax again.
        """
        probs = torch.stack([
            torch.softmax(model(x), dim=1)  # Get the probabilities for each model.  # noqa
            for model in self.models
        ])  # Tuple with (n_models, B, 2)

        # Soft-voting.
        weights = self.weights.to(x.device).view(-1, 1, 1)
        avg_probs = (probs * weights).sum(dim=0)

        return torch.log(avg_probs + 1.e-12)
