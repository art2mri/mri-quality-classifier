"""Trainer class."""

from pathlib import Path

import pandas as pd

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader

from src.evaluation.metrics import (
    compute_metrics,
    scalar_metrics,
    ClassificationMetrics
)


class MRIQualityTrainer:
    """Trainer for MRI quality classification models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str | torch.device,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
        monitor_metric: str = 'auc'
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = Path(checkpoint_dir)
        self.monitor_metric = monitor_metric
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        self.best_score = float('-inf')

    def train_one_epoch(
        self,
        log_every: int = 10,
        verbose: bool = False
    ) -> float:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        if verbose:
            print('[TRAIN] Starting training epoch.')

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)

            self.optimizer.zero_grad()  # Zero grads for every batch.
            logits = self.model(inputs)  # Make preds for every batch.
            loss = self.criterion(logits, targets)  # Compute loss.
            loss.backward()  # Compute gradients.
            self.optimizer.step()  # Adjust weights.

            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            if verbose and (batch_idx == 1 or batch_idx % log_every == 0):
                print(
                    f"[TRAIN] Batch {batch_idx:04d} | "
                    f"loss={batch_loss:.4f} | "
                    f"running_mean_loss={total_loss / num_batches:.4f}"
                )

        mean_loss = total_loss / max(num_batches, 1)

        if verbose:
            print(f"[TRAIN] Epoch finished | mean_loss={mean_loss:.4f}")

        return mean_loss

    @torch.no_grad()
    def validate(
        self,
        log_every: int = 10,
        verbose: bool = False
    ) -> tuple[float, ClassificationMetrics]:
        """
        Run one full validation epoch.

        It returns a tuple containing the mean validation loss across batches
        and a dict with validation metrics, such as acc, prec, rec, f1, auc,
        sensitivity, and specificity.
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_targets = []
        all_predictions = []
        all_scores = []

        if verbose:
            print('[VAL] Starting validation.')

        for batch_idx, batch in enumerate(self.val_loader, start=1):
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            scores = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            num_batches += 1

            all_targets.append(targets.cpu())
            all_predictions.append(predictions.cpu())
            all_scores.append(scores.cpu())

            if verbose and (batch_idx == 1 or batch_idx % log_every == 0):
                print(
                    f"[VAL] Batch {batch_idx:04d} | "
                    f"loss={loss.item():.4f} | "
                    f"running_mean_loss={total_loss / num_batches:.4f}"
                )

        y_true = torch.cat(all_targets).numpy()
        y_pred = torch.cat(all_predictions).numpy()
        y_score = torch.cat(all_scores).numpy()

        metrics = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score
        )
        mean_loss = total_loss / max(num_batches, 1)

        if verbose:
            print(f"[VAL] Finished | mean_loss={mean_loss:.4f} | metrics={metrics}")

        return mean_loss, metrics

    def fit(
        self,
        num_epochs: int,
        batch_verbose: bool = False,
        verbose: bool = False
    ) -> dict:
        """
        Train the model for multiple epochs and save the best checkpoint.

        Args:
            num_epochs: number of max epochs.
            batch_verbose: whether to print batch-level training progress.
        """
        if verbose:
            print(f"[FIT] Starting training for {num_epochs} epochs.")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            if verbose:
                print(f"[FIT] Epoch {epoch}/{num_epochs}")

            train_loss = self.train_one_epoch(verbose=batch_verbose)
            val_loss, metrics = self.validate(verbose=batch_verbose)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)

            self.writer.add_scalars(
                'loss', {'train': train_loss, 'val': val_loss}, epoch
            )
            for name, value in scalar_metrics(metrics).items():
                self.writer.add_scalar(f"val/{name}", value, epoch)

            score = metrics[self.monitor_metric]

            if score > self.best_score:
                self.best_score = score
                torch.save(
                    obj=self.model.state_dict(),
                    f=self.checkpoint_dir / f"{self.model._get_name()}_best_model.pt"
                )

                if verbose:
                    print(
                        f"[FIT] New best model saved | "
                        f"{self.monitor_metric}={score:.4f}"
                    )

            if verbose:
                print(
                    f"[FIT] Epoch {epoch} done | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"{self.monitor_metric}={score:.4f}"
                )

        self.writer.close()

        # Save final csv with losses and metrics.
        self._save_csv(verbose=verbose)

        if verbose:
            print('[FIT] Training finished.')

        return self.history

    @torch.no_grad()
    def predict(self, dataloader: DataLoader, verbose: bool = False):
        """
        Generate class predictions and positive-class scores for a dataloader.
        """
        self.model.eval()

        all_predictions = []
        all_scores = []

        if verbose:
            print('[PRED] Starting prediction.')

        for batch in dataloader:
            inputs = batch['image'].to(self.device)

            logits = self.model(inputs)
            scores = torch.softmax(logits, dim=1)[:, 1]
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu())
            all_scores.append(scores.cpu())

        predictions = torch.cat(all_predictions)
        scores = torch.cat(all_scores)

        if verbose:
            print('[PRED] Prediction finished.')

        return predictions, scores

    def _save_csv(self, verbose: bool = False) -> None:
        rows = [
            {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, **scalar_metrics(metrics)}
            for epoch, (train_loss, val_loss, metrics) in enumerate(
                zip(
                    self.history['train_loss'],
                    self.history['val_loss'],
                    self.history['metrics']
                ),
                start=1
            )
        ]

        pd.DataFrame(rows).to_csv(
            self.checkpoint_dir / f'_{self.model._get_name()}_history.csv',
            index=False
        )

        if verbose:
            print('[SAV] .csv file saved on disk.')
