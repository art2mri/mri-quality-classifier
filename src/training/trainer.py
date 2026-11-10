"""Training module."""

import random
from pathlib import Path

import numpy as np
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
    Trainer class for MRI quality classification models. It set up model, train
    and val data loaders, optimizer, loss, logging, and checkpoint.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function.
        checkpoint_dir: Directory where checkpoints will be saved.
        log_dir: Directory for TensorBoard logs.
        accumulation_steps: Gradient steps to simulate larger batch sizes. `1`
            means no accumulation.
        scheduler: Optional learning rate scheduler, stepped with loss_val
            every epoch.
        grad_clip_norm: Optional max gradient norm to clip exploding gradients
            before each optimizer step when set.
        device: Compute device, e.g., 'cpu' or 'cuda'.
        monitor_metric: Metric used to track and save the best model. `f1` by
            default.
        seed: Seed for torch/numpy/random, for reproducible runs.
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
        grad_clip_norm: float | None = None,
        device: str | torch.device = 'cpu',
        monitor_metric: str = 'f1',
        seed: int = 42
    ) -> None:
        self._set_seed(seed=seed)
        self.device = torch.device(device=device)
        self.model = model.to(device=self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.monitor_metric = monitor_metric
        self.checkpoint_dir = Path(checkpoint_dir)
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self.best_score = float('-inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }

    def _set_seed(self, seed: int) -> None:
        """Seed torch/numpy/random and disable cuDNN non-determinism."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _save_weights(self, suffix: str) -> Path:
        """Save the current model weights."""
        checkpoint_path = self.checkpoint_dir / f"{self.model._get_name()}_{suffix}.pt"  # noqa
        torch.save(obj=self.model.state_dict(), f=checkpoint_path)

        return checkpoint_path

    def train_one_epoch(
        self,
        log_every: int = 10,
        verbose: bool = False
    ) -> float:
        """Run one training epoch and returns the train loader mean loss."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        num_train_batches = len(self.train_loader)

        if verbose:
            print('[TRAIN] Starting training epoch.')

        self.optimizer.zero_grad()  # Zero grads once, before accumulating.

        # Batch training loop. It calculates the loss for each batch.
        for batch_idx, batch in enumerate(self.train_loader, start=1):
            inputs = batch['image'].to(self.device)
            targets = batch['label'].to(self.device)

            logits = self.model(inputs)  # Batch prediction.
            loss = self.criterion(logits, targets)  # Compute loss.

            # Scaled backward to accumulate loss.
            (loss / self.accumulation_steps).backward()

            is_last_batch = batch_idx == num_train_batches

            if (batch_idx % self.accumulation_steps == 0) or is_last_batch:
                if self.grad_clip_norm is not None:
                    # Caps the global grad norm, guarding against exploding grads.  # noqa
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.grad_clip_norm
                    )

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
            print(
                f"[VAL] Finished | "
                f"mean_loss={mean_loss:.4f} | "
                f"metrics={metrics}"
            )

        return mean_loss, metrics

    def fit(
        self,
        num_epochs: int,
        patience: int | None = None,
        log_every: int = 10,
        batch_verbose: bool = False,
        verbose: bool = False
    ) -> dict:
        """
        Train the model for multiple epochs, saving both the best and the last
        checkpoints.

        Args:
            num_epochs: Number of max epochs.
            patience: Stop early after this many epochs without improvement in
                `monitor_metric`. None disables early stopping.
        """
        if verbose:
            print(f"[FIT] Starting training for {num_epochs} epochs.")

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        epochs_without_improvement = 0

        # Training loop.
        for epoch in range(1, num_epochs + 1):
            if verbose:
                print(f"[FIT] Epoch {epoch}/{num_epochs}")

            # One single epoch passes over the train loader using batches and
            # validates over the entire val dataloader.
            train_loss = self.train_one_epoch(log_every=log_every, verbose=batch_verbose)  # noqa
            val_loss, metrics = self.validate(log_every=log_every, verbose=batch_verbose)  # noqa

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['metrics'].append(metrics)

            # Lower the learning rate if the val_loss stalls.
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            # TensorBoard logs.
            self.writer.add_scalars(
                'loss',
                {'train': train_loss, 'val': val_loss},
                epoch
            )
            for name, value in scalar_metrics(metrics=metrics).items():
                self.writer.add_scalar(f"val/{name}", value, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)  # noqa

            score = metrics[self.monitor_metric]  # Get the epoch score.

            # Checkpointing and early-stopping bookkepping.
            if score > self.best_score:
                self.best_score = score
                epochs_without_improvement = 0

                best_path = self._save_weights(suffix='best_model')

                if verbose:
                    print(
                        f"[FIT] New best model saved to {best_path} | "
                        f"{self.monitor_metric}={score:.4f}"
                    )
            else:
                epochs_without_improvement += 1

            # Always overwrite the last-epoch checkpoint.
            _ = self._save_weights(suffix='last_model')

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

        if verbose:
            print('[FIT] Training finished.')

        return self.history
