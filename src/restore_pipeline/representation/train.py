"""Architecture-agnostic training loop for representation models.

The loop only needs an `nn.Module` and a `loss_fn(model, batch) -> loss`.
Anything model-specific (what forward returns, how the loss is built) lives
in the loss function, so a new model type only needs a new `loss_fn`.

    history = fit(model, train_dl, val_dl, TrainConfig(epochs=10))
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from restore_pipeline.representation.models import MODEL_REGISTRY, RepresentationModel

#
# CONFIG
#

@dataclass(frozen=True)
class TrainConfig:
    """
    Hyperparameters of a training run. Model architecture args do NOT go here.

    Attributes:
        epochs: Number of passes over the training set.
        lr: Adam learning rate.
        weight_decay: Adam weight decay (L2 penalty). 0 disables it.
        grad_clip: Max gradient norm, or None to disable clipping.
        device: Device the model and batches are moved to.
        show_progress: Show tqdm progress bars. Turn off for scripts or logs.
        checkpoint_path: If set, `fit` saves the model with the lowest
            validation loss here. None disables checkpointing.
    """

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    show_progress: bool = True
    checkpoint_path: str | None = None


#
# LOSS FUNCTIONS: the only place that knows what the model outputs
#

# (model, batch) -> scalar loss tensor
LossFn = Callable[[nn.Module, dict], torch.Tensor]

# (epoch, train_loss, val_loss) -> None; called by `fit` after every epoch
EpochCallback = Callable[[int, float, float], None]


def reconstruction_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    """Plain autoencoder: MSE between the reconstruction and the input image."""
    device = next(model.parameters()).device
    x = batch["image"].to(device, non_blocking=True)
    return F.mse_loss(model(x), x)

def damaged_reconstruction_loss(model: nn.Module, batch: dict) -> torch.Tensor:
    """Plain autoencoder: MSE between the reconstruction and the input image."""
    device = next(model.parameters()).device
    x = batch["image"].to(device, non_blocking=True)
    y = batch["target"].to(device, non_blocking=True)
    return F.mse_loss(model(x), y)


#
# STEPS
#

def train_step(
    model: nn.Module,
    batch: dict,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
) -> float:
    """One optimization step on a single batch. Returns the batch loss."""
    loss = loss_fn(model, batch)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return loss.item()


@torch.inference_mode()
def val_step(model: nn.Module, batch: dict, loss_fn: LossFn) -> float:
    """Loss on a single batch with gradients disabled. Returns the batch loss."""
    return loss_fn(model, batch).item()


#
# EPOCHS
#

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    grad_clip: float | None = None,
    show_progress: bool = True,
) -> float:
    """Train for one epoch. Returns the mean loss per sample."""
    model.train()
    total_loss = 0.0
    n_samples = 0

    progress = tqdm(loader, desc="Training", leave=False, disable=not show_progress)
    for batch in progress:
        batch_size = batch["image"].size(0)
        batch_loss = train_step(model, batch, loss_fn, optimizer, grad_clip)

        # Weight by batch size so a smaller last batch does not skew the mean.
        total_loss += batch_loss * batch_size
        n_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / n_samples:.4f}")

    return total_loss / n_samples


def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: LossFn,
    show_progress: bool = True,
) -> float:
    """Evaluate for one epoch. Returns the mean loss per sample."""
    model.eval()
    total_loss = 0.0
    n_samples = 0

    progress = tqdm(loader, desc="Validation", leave=False, disable=not show_progress)
    for batch in progress:
        batch_size = batch["image"].size(0)
        batch_loss = val_step(model, batch, loss_fn)

        total_loss += batch_loss * batch_size
        n_samples += batch_size
        progress.set_postfix(loss=f"{total_loss / n_samples:.4f}")

    return total_loss / n_samples


def run_epoch(
    model: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> tuple[float, float]:
    """Train over `train_dl`, then evaluate over `val_dl`. Returns (train_loss, val_loss)."""
    train_loss = train_epoch(model, train_dl, loss_fn, optimizer, cfg.grad_clip, cfg.show_progress)
    val_loss = val_epoch(model, val_dl, loss_fn, cfg.show_progress)
    return train_loss, val_loss


#
# CHECKPOINTS
#

def save_checkpoint(
    model: RepresentationModel,
    path: str | Path,
    history: dict[str, list[float]] | None = None,
) -> None:
    """
    Save weights together with what is needed to rebuild the model.

    Stores the class name (looked up in `MODEL_REGISTRY` on load), the
    constructor kwargs and the loss history alongside the state dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": type(model).__name__,
            "init_kwargs": model.init_kwargs,
            "state_dict": model.state_dict(),
            "history": history,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> RepresentationModel:
    """
    Rebuild a model from a checkpoint made by `save_checkpoint`.

    Returns the model on `device`, in eval mode.

    Raises:
        KeyError: If the saved class is not in `MODEL_REGISTRY`.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    class_name = checkpoint["model_class"]
    if class_name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model class '{class_name}' is not in MODEL_REGISTRY "
            f"(known: {sorted(MODEL_REGISTRY)}). Register it in representation_models.py."
        )

    model = MODEL_REGISTRY[class_name](**checkpoint["init_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


#
# FIT
#

def fit(
    model: RepresentationModel,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: TrainConfig = TrainConfig(),
    loss_fn: LossFn = reconstruction_loss,
    on_epoch_end: EpochCallback | None = None,
) -> dict[str, list[float]]:
    """
    Train `model` for `cfg.epochs` epochs with Adam.

    If `cfg.checkpoint_path` is set, the model is saved there whenever the
    validation loss improves, so the file holds the best epoch, not the last.
    The history stored in it only covers epochs up to that point.

    `on_epoch_end` is called after each epoch (and after any checkpoint save),
    which is the hook for experiment trackers such as MLflow or Comet.

    Args:
        model (RepresentationModel): Model to train. Moved to `cfg.device`.
        train_dl (DataLoader): Training batches.
        val_dl (DataLoader): Validation batches.
        cfg (TrainConfig): Hyperparameters and progress-bar setting.
        loss_fn (LossFn): Turns (model, batch) into a scalar loss.
        on_epoch_end (EpochCallback | None): Called as
            `on_epoch_end(epoch, train_loss, val_loss)` with a 0-based epoch.

    Returns:
        dict[str, list[float]]: Per-epoch mean losses under 'train' and 'val'.
    """
    model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: dict[str, list[float]] = {"train": [], "val": []}
    best_val_loss = float("inf")

    for epoch in tqdm(range(cfg.epochs), desc="Epochs", disable=not cfg.show_progress):
        train_loss, val_loss = run_epoch(model, train_dl, val_dl, loss_fn, optimizer, cfg)
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if cfg.checkpoint_path is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, cfg.checkpoint_path, history)

        if on_epoch_end is not None:
            on_epoch_end(epoch, train_loss, val_loss)

    return history
