"""Trainer class."""

import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader

from src.evaluation.metrics import (
    scalar_metrics,
    compute_metrics,
    ClassificationMetrics
)


class MRIQualityTrainer:
    """
    Trainer class for MRI quality classification models. It set up model, data
    loaders, optimizer, loss, logging, and checkpoint.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function.
        checkpoint_dir: Directory where checkpoints will be saved.
        log_dir: Directory for TensorBoard logs.
        accumulation_steps: Gradient steps to simulate larger batch sizes.
        scheduler: Optional learning rate scheduler, stepped with loss_val
            every epoch.
        device: Compute device, e.g., 'cpu' or 'cuda'.
        monitor_metric: Metric used to track and save the best model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        criterion: nn.Module,
        checkpoint_dir: str | Path,
        log_dir: str | Path,
        accumulation_steps: int = 1,
        scheduler: ReduceLROnPlateau | None = None,
        device: str | torch.device = 'cpu',
        monitor_metric: str = 'auc'
    ) -> None:
        self.device = torch.device(device=device)
        self.model = model.to(device=self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.scheduler = scheduler
        self.monitor_metric = monitor_metric
        self.checkpoint_dir = Path(checkpoint_dir)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.best_score = float('-inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }

    def train_one_epoch(self, log_every: int = 10, verbose: bool = False) -> float:
        """Run one training epoch with gradient accumulation steps."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        num_train_batches = len(self.train_loader)

        if verbose:
            print('[TRAIN] Starting training epoch.')

        self.optimizer.zero_grad()  # Zero grads once, before accumulating.

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)

            logits = self.model(inputs)  # Batch prediction.
            loss = self.criterion(logits, targets)  # Compute loss.

            # Scaled backward to accumulate loss.
            (loss / self.accumulation_steps).backward()

            is_last_batch = batch_idx == num_train_batches

            if (batch_idx % self.accumulation_steps == 0) or is_last_batch:
                self.optimizer.step()  # Apply accumulated grads.
                self.optimizer.zero_grad()

            batch_loss = loss.item()  # Unscaled, for interpretable logging.
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
        Run one full validation epoch. It returns a tuple containing the mean
        validation loss across batches and a dict with validation metrics.
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

            scores = torch.softmax(logits, dim=1)[:, 1]  # Probabilities.
            predictions = torch.argmax(logits, dim=1)  # Final class.

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
        patience: int | None = None,
        batch_verbose: bool = False,
        verbose: bool = False
    ) -> dict:
        """
        Train the model for multiple epochs and save the best checkpoint.

        Args:
            num_epochs: number of max epochs.
            patience: stop early after this many epochs without improvement
                in `monitor_metric`. None disables early stopping.
        """
        if verbose:
            print(f"[FIT] Starting training for {num_epochs} epochs.")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_without_improvement = 0

        # Training loop.
        for epoch in range(1, num_epochs + 1):
            if verbose:
                print(f"[FIT] Epoch {epoch}/{num_epochs}")

            # One single epoch passes over the entire train dataloader. After
            # that, it validates over the entire val dataloader.
            train_loss = self.train_one_epoch(verbose=batch_verbose)
            val_loss, metrics = self.validate(verbose=batch_verbose)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)

            # Scheduler reacts to val_loss and it may lower the learning rate
            # if it stalls.
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # Add TensorBoard logs.
            self.writer.add_scalars(
                'loss',
                {'train': train_loss, 'val': val_loss},
                epoch
            )
            for name, value in scalar_metrics(metrics=metrics).items():
                self.writer.add_scalar(f"val/{name}", value, epoch)
            self.writer.add_scalar(
                'lr', self.optimizer.param_groups[0]['lr'], epoch
            )

            score = metrics[self.monitor_metric]  # Get the epoch score.

            # Checkpointing and early-stopping bookkeeping.
            if score > self.best_score:
                self.best_score = score
                epochs_without_improvement = 0

                torch.save(
                    obj=self.model.state_dict(),
                    f=self.checkpoint_dir / f"{self.model._get_name()}_best_model.pt"
                )

                if verbose:
                    print(
                        f"[FIT] New best model saved | "
                        f"{self.monitor_metric}={score:.4f}"
                    )
            else:
                epochs_without_improvement += 1

            if verbose:
                print(
                    f"[FIT] Epoch {epoch} done | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"{self.monitor_metric}={score:.4f}"
                )

            # Stop once too many epochs pass without a new best score.
            if patience is not None and epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"[FIT] Early stopping at epoch {epoch} "
                        f"(no improvement in {patience} epochs)."
                    )

                break

        # Close TensorBoard to ensure all info will be saved on disk.
        self.writer.close()

        # I WILL SAVE SOME LOG DATASET HERE.

        if verbose:
            print('[FIT] Training finished.')

        return self.history

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        verbose: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate class predictions and positive-class scores for a given
        dataloader.
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

    def save_dict(self, verbose: bool = False) -> None:
        pass
