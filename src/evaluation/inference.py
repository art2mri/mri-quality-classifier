# type: ignore
"""Inference module."""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from monai.data import DataLoader

from src.models.model import Models, get_model
from src.evaluation.metrics import ClassificationMetrics, compute_metrics


def load_model(
    checkpoint_path: str | Path,
    model_name: Models,
    spatial_dims: Literal[2, 3] = 3,
    device: str | torch.device = 'cpu'
) -> nn.Module:
    """Build a model and load trained weights from a checkpoint file."""
    device = torch.device(device=device)

    print(f"[LOAD_MODEL] Loading {checkpoint_path} on {device}.")

    # Create an empty model with the same architecture used in training.
    model = get_model(model_name=model_name, spatial_dims=spatial_dims)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict=state_dict)

    print('[LOAD_MODEL] Loaded.')

    return model.to(device=device)


@torch.no_grad()
def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = 'cpu',
    sample_ids: list[str] | None = None,
    verbose: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Run a trained model over a dataloader and return predicted class per
    sample, the probs, and the ground truth labels (if exists).

    Args:
        model: Model with weights already loaded.
        dataloader: DataLoader to run inference on. Use `shuffle=False` to
            match results with `sample_ids`.
        device: Compute device, e.g., 'cpu' or 'cuda'.
        sample_ids: Optional identifiers (e.g., image paths).
    """
    device = torch.device(device=device)
    model = model.to(device=device)
    model.eval()

    predictions, scores, targets = [], [], []
    offset = 0  # Index of the current batch first sample.

    if verbose:
        print('[PRED] Starting prediction.')

    for batch in dataloader:
        inputs = batch['image'].to(device)
        labels = batch.get('label')  # None for unlabeled data.
        batch_size = inputs.shape[0]

        logits = model(inputs)
        batch_scores = torch.softmax(logits, dim=1)[:, 1]
        batch_predictions = torch.argmax(logits, dim=1)

        if verbose:
            if sample_ids is not None:
                ids = sample_ids[offset:offset + batch_size]
            else:
                ids = range(offset, offset + batch_size)

            for i in range(batch_size):
                label = labels[i].item() if labels is not None else 'N/A'
                print(
                    f"[PRED] {ids[i]} | "
                    f"label={label} | "
                    f"prediction={batch_predictions[i].item()} | "
                    f"score={batch_scores[i].item():.4f}"
                )

        # Move results to CPU.
        predictions.append(batch_predictions.cpu())
        scores.append(batch_scores.cpu())

        if labels is not None:
            targets.append(labels.cpu())

        offset += batch_size

    if verbose:
        print('[PRED] Prediction finished.')

    return (
        torch.cat(predictions),
        torch.cat(scores),
        torch.cat(targets) if targets else None
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = 'cpu',
    sample_ids: list[str] | None = None,
    verbose: bool = True
) -> ClassificationMetrics:
    """Compute classification metrics for a labeled dataloader."""
    predictions, scores, targets = predict(
        model=model,
        dataloader=dataloader,
        device=device,
        sample_ids=sample_ids,
        verbose=verbose
    )

    if targets is None:
        raise ValueError("'evaluate' requires labels in the dataloader.")

    metrics = compute_metrics(
        y_true=targets.numpy(),
        y_pred=predictions.numpy(),
        y_score=scores.numpy()
    )

    if verbose:
        print(f"[EVAL] Metrics: {metrics}")

    return metrics
