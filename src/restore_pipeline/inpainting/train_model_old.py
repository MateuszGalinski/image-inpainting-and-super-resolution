from pathlib import Path
from time import perf_counter
from typing import Callable

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def train_model(
    model: nn.Module,
    opt: Optimizer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    criterion: Callable[[Tensor, Tensor], Tensor],
    epochs: int = 20,
    *,
    save_path: str | Path = "model_best.pt",
    use_amp: bool = True,
) -> dict[str, list[float]]:
    device = next(model.parameters()).device
    amp_enabled = use_amp and device.type == "cuda"

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    best_epoch = 0

    print(f"Device: {device}")
    print(f"Mixed precision: {amp_enabled}")
    print(f"Training batches: {len(train_dl)}")
    print(f"Validation batches: {len(val_dl)}")
    print(f"Epochs: {epochs}\n", flush=True)

    def run_epoch(
        loader: DataLoader,
        training: bool,
        desc: str,
    ) -> float:
        model.train(training)

        total_loss = torch.zeros((), device=device)
        total_images = 0

        progress = tqdm(
            loader,
            desc=desc,
            leave=True,
            dynamic_ncols=True,
        )

        with torch.set_grad_enabled(training):
            for batch in progress:
                damaged = batch["image"].to(
                    device, non_blocking=True
                )
                clean = batch["target"].to(
                    device, non_blocking=True
                )

                if training:
                    opt.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    restored = model(damaged)
                    loss = criterion(restored, clean)

                if training:
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    else:
                        loss.backward()
                        opt.step()

                batch_size = damaged.size(0)

                total_loss += loss.detach() * batch_size
                total_images += batch_size

        return (total_loss / total_images).item()

    # Training
    for epoch in range(1, epochs + 1):
        start = perf_counter()

        train_loss = run_epoch(
            train_dl,
            training=True,
            desc=f"Epoch {epoch}/{epochs} [Train]",
        )

        val_loss = run_epoch(
            val_dl,
            training=False,
            desc=f"Epoch {epoch}/{epochs} [Val]",
        )

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        # Save best checkpoint immediately
        improved = val_loss < best_val_loss

        if improved:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(model.state_dict(), save_path)

        elapsed = perf_counter() - start

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train L1: {train_loss:.4f} | "
            f"Val L1: {val_loss:.4f} | "
            f"Time: {elapsed:.1f}s"
            + (" | Best checkpoint saved!" if improved else ""),
            flush=True,
        )

    # Load best checkpoint
    if best_epoch == 0:
        raise RuntimeError("No checkpoint was selected.")

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device,
            weights_only=True,
        )
    )

    print(
        f"\nBest epoch: {best_epoch} "
        f"| Validation L1: {best_val_loss:.4f}"
    )
    print(f"Loaded best checkpoint from {save_path}")

    # Final evaluation
    test_loss = run_epoch(
        test_dl,
        training=False,
        desc="Testing",
    )

    print(f"Test L1: {test_loss:.4f}")

    return history