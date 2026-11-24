import os
import random

from pathlib import Path
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

from .losses import _region_errors, hybrid_inpainting_loss


class _FixedBatchSampler(Sampler[list[int]]):
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def train_model(
    model: nn.Module,
    opt: Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    criterion: Callable[[Tensor, Tensor, Tensor], Tensor] = hybrid_inpainting_loss,
    epochs: int = 20,
    *,
    save_path: str | Path = "model_best.pt",
    use_amp: bool = True,
    resume: bool = True,
    checkpoint_every: int = 100,
) -> dict[str, list[float]]:

    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")

    device = next(model.parameters()).device
    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    save_path = Path(save_path)
    resume_path = save_path.with_name(
        f"{save_path.stem}_resume{save_path.suffix}"
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": 1,
        "phase": "train",
        "batches": None,
        "next_batch": 0,
        "loss_sum": 0.0,
        "images_seen": 0,
        "train_loss": None,
        "history": {"train": [], "val": []},
        "best_val_loss": float("inf"),
        "best_epoch": None,
    }

    def atomic_save(obj, path):
        tmp = path.with_name(path.name + ".tmp")
        torch.save(obj, tmp)
        os.replace(tmp, path)

    # Save everything needed to resume training
    def save_recovery():
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "state": state,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if device.type == "cuda"
                    else None
                ),
                "python_rng": random.getstate(),
                "loader_rng": (
                    train_dl.generator.get_state()
                    if train_dl.generator is not None
                    else None
                ),
            },
            resume_path,
        )

    # Restore previous session
    if resume and resume_path.exists():
        ckpt = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        state = ckpt["state"]

        torch.set_rng_state(ckpt["torch_rng"].cpu())

        if device.type == "cuda" and ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

        random.setstate(ckpt["python_rng"])

        if train_dl.generator is not None and ckpt["loader_rng"] is not None:
            train_dl.generator.set_state(ckpt["loader_rng"].cpu())

        print(
            f"Resumed: epoch {state['epoch']}, "
            f"phase={state['phase']}, "
            f"batch={state['next_batch']}",
            flush=True,
        )

    print(f"Device: {device}")
    print(f"Mixed precision: {amp_enabled}")
    print(f"Training batches: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")
    print(f"Epochs: {epochs}")
    print(f"Checkpoint interval: {checkpoint_every} batches")
    print(f"Recovery checkpoint: {resume_path}\n", flush=True)

    # Build a loader for the remaining batches of an epoch
    def remaining_loader(batches):
        kwargs = {
            "dataset": train_dl.dataset,
            "batch_sampler": _FixedBatchSampler(batches),
            "num_workers": train_dl.num_workers,
            "collate_fn": train_dl.collate_fn,
            "pin_memory": train_dl.pin_memory,
            "worker_init_fn": train_dl.worker_init_fn,
            "timeout": train_dl.timeout,
            "generator": train_dl.generator,
        }

        if train_dl.num_workers > 0:
            kwargs["prefetch_factor"] = train_dl.prefetch_factor
            kwargs["persistent_workers"] = train_dl.persistent_workers

        return DataLoader(**kwargs)

    # Validation and testing use MASKED L1 ONLY
    def masked_evaluate(loader, desc):
        model.eval()

        total_error = torch.zeros((), device=device)
        total_images = torch.zeros((), device=device)

        with torch.inference_mode():
            for batch in tqdm(loader, desc=desc):
                damaged = batch["image"].to(
                    device, non_blocking=True
                )
                clean = batch["target"].to(
                    device, non_blocking=True
                )
                mask = batch["mask"].to(
                    device, non_blocking=True
                )

                with torch.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    restored = model(damaged)

                    hole_l1, _, has_hole = _region_errors(
                        restored, clean, mask
                    )

                total_error += hole_l1.sum()
                total_images += has_hole.sum()

        if total_images.item() == 0:
            raise RuntimeError(
                "Validation/test data contain no damaged pixels."
            )

        return (total_error / total_images).item()

    # Training
    for epoch in range(state["epoch"], epochs + 1):
        start = perf_counter()

        if state["phase"] == "train":

            # Generate batch order once per epoch
            if state["batches"] is None:
                state["batches"] = list(train_dl.batch_sampler)
                state["next_batch"] = 0
                state["loss_sum"] = 0.0
                state["images_seen"] = 0

                save_recovery()

            batches = state["batches"]

            running_loss = torch.tensor(
                state["loss_sum"],
                device=device,
            )

            model.train()

            with tqdm(
                total=len(batches),
                initial=state["next_batch"],
                desc=f"Epoch {epoch}/{epochs} [Train]",
                dynamic_ncols=True,
            ) as progress:

                for batch in remaining_loader(
                    batches[state["next_batch"]:]
                ):
                    damaged = batch["image"].to(
                        device, non_blocking=True
                    )
                    clean = batch["target"].to(
                        device, non_blocking=True
                    )
                    mask = batch["mask"].to(
                        device, non_blocking=True
                    )

                    opt.zero_grad(set_to_none=True)

                    with torch.autocast(
                        device_type=device.type,
                        enabled=amp_enabled,
                    ):
                        restored = model(damaged)

                        # HYBRID loss for training
                        loss = criterion(restored, clean, mask)

                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()

                    batch_size = damaged.size(0)

                    running_loss += loss.detach() * batch_size
                    state["images_seen"] += batch_size
                    state["next_batch"] += 1

                    progress.update(1)

                    # Periodic recovery checkpoint
                    if state["next_batch"] % checkpoint_every == 0:
                        state["loss_sum"] = running_loss.item()
                        save_recovery()

            state["train_loss"] = (
                running_loss / state["images_seen"]
            ).item()

            state["phase"] = "val"
            state["loss_sum"] = running_loss.item()

            # Save completed training epoch
            save_recovery()

        # MASKED L1 for validation
        val_loss = masked_evaluate(
            val_dl,
            f"Epoch {epoch}/{epochs} [Val masked L1]",
        )

        state["history"]["train"].append(state["train_loss"])
        state["history"]["val"].append(val_loss)

        # Select best weights using MASKED validation L1
        improved = val_loss < state["best_val_loss"]

        if improved:
            state["best_val_loss"] = val_loss
            state["best_epoch"] = epoch

            atomic_save(model.state_dict(), save_path)

        elapsed = perf_counter() - start

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train hybrid: {state['train_loss']:.4f} | "
            f"Val masked L1: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
            + (" | Best saved!" if improved else ""),
            flush=True,
        )

        # Prepare checkpoint for the next epoch
        state.update(
            epoch=epoch + 1,
            phase="train",
            batches=None,
            next_batch=0,
            loss_sum=0.0,
            images_seen=0,
            train_loss=None,
        )

        save_recovery()

    # Load best model for testing
    if not save_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True,
        )
    )

    print(
        f"\nBest epoch: {state['best_epoch']} | "
        f"Best masked val L1: {state['best_val_loss']:.4f}"
    )

    # Final evaluation uses MASKED L1
    test_loss = masked_evaluate(
        test_dl,
        "Test masked L1",
    )

    print(f"Test masked L1: {test_loss:.4f}")

    return state["history"]