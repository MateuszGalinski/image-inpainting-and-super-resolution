"""Pliki eksperymentów i odtwarzanie stanu ukończonej epoki."""

from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import json
import platform
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .models import build_model


def environment_info(device: torch.device) -> dict[str, object]:
    """Opisz biblioteki i urządzenie istotne dla odtwarzania treningu."""

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "pillow": version("pillow"),
        "datasets": version("datasets"),
        "torchvision": version("torchvision"),
        "device_type": device.type,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "cuda_devices": torch.cuda.device_count() if device.type == "cuda" else 0,
        "cpu_threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }


def seed_everything(seed: int, device: torch.device) -> None:
    """Ustaw początek losowania i powtarzalny wybór algorytmów cuDNN."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    """Zastąp CSV dopiero po zapisaniu całej tabeli do pliku tymczasowego."""

    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def create_run(base_path: Path, config: dict[str, object]) -> Path:
    """Utwórz nowy katalog z konfiguracją bez nadpisywania istniejącego treningu."""

    architecture = config["architecture"]
    scale = architecture["params"]["scale_factor"]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    label = config["experiment_name"]
    name = f"{label}_x{scale}_mse_{stamp}"

    base_path.mkdir(parents=True, exist_ok=True)
    run_path = base_path / name
    run_path.mkdir(exist_ok=False)

    temporary = run_path / "config.json.tmp"
    temporary.write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(run_path / "config.json")

    return run_path


def load_run_config(run_path: Path) -> dict[str, object]:
    """Wczytaj ustawienia uruchomienia w nowym formacie checkpointów."""

    config = json.loads((run_path / "config.json").read_text(encoding="utf-8"))

    if config.get("format_version") != 1:
        raise ValueError(
            "Nieobsługiwana wersja konfiguracji; stare pliki nie są migrowane."
        )

    return config


def capture_rng_state(
    loaders: tuple[DataLoader, DataLoader], device: torch.device
) -> dict[str, object]:
    """Zachowaj losowanie po epoce, przed ponownym utworzeniem workerów."""

    numpy_state = np.random.get_state()

    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "loaders": [loader.generator.get_state() for loader in loaders],
    }


def restore_rng_state(
    state: dict[str, object],
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
) -> None:
    """Przywróć losowanie dopiero po utworzeniu modelu i loaderów."""

    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])

    if device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda"])

    for loader, generator_state in zip(loaders, state["loaders"], strict=True):
        loader.generator.set_state(generator_state)


def save_checkpoint(path: Path, checkpoint: dict[str, object]) -> None:
    """Podmień checkpoint dopiero po ukończeniu zapisu nowej wersji."""

    temporary = path.with_name(path.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def read_checkpoint(path: Path) -> dict[str, object]:
    """Odczytaj nowy format i odrzuć nieobsługiwane starsze checkpointy."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if checkpoint.get("format_version") != 1:
        raise ValueError("Nieobsługiwany checkpoint; wymagany format_version=1.")
    if checkpoint["architecture"] != checkpoint["config"]["architecture"]:
        raise ValueError("Opis architektury nie zgadza się z konfiguracją.")

    epoch = checkpoint["epoch"]
    history = checkpoint["history"]

    if epoch < 1 or not 1 <= checkpoint["best_epoch"] <= epoch:
        raise ValueError("Niepoprawna numeracja epok w checkpointcie.")
    if any(
        len(history[key]) != epoch
        for key in ("train_loss", "val_loss", "learning_rate", "epoch_seconds")
    ):
        raise ValueError("Historia nie odpowiada liczbie zapisanych epok.")

    return checkpoint


def initial_training_state() -> dict[str, object]:
    """Zacznij nową historię i wybór najlepszej epoki."""

    return {
        "epoch": 0,
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "history": {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
            "epoch_seconds": [],
        },
    }


def record_epoch(
    state: dict[str, object],
    train_loss: float,
    val_loss: float,
    learning_rate: float,
    epoch_seconds: float,
) -> None:
    """Dopisz straty, użyty LR i czas treningu z walidacją bez zapisu plików."""

    if not np.isfinite(train_loss) or not np.isfinite(val_loss):
        raise ValueError("Strata nie jest skończona; sprawdź dane i trening.")

    for key, value in (
        ("train_loss", train_loss),
        ("val_loss", val_loss),
        ("learning_rate", learning_rate),
        ("epoch_seconds", epoch_seconds),
    ):
        state["history"][key].append(value)

    state["epoch"] += 1

    if val_loss < state["best_val_loss"]:
        state["best_val_loss"] = val_loss
        state["best_epoch"] = state["epoch"]


def save_history(history: dict[str, list[float]], run_path: Path) -> None:
    """Wyeksportuj wszystkie ukończone epoki do jednej czytelnej tabeli."""

    frame = pd.DataFrame(history)
    frame.insert(0, "epoch", np.arange(1, len(frame) + 1))
    save_csv(frame, run_path / "history.csv")


def save_epoch(
    run_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    config: dict[str, object],
    state: dict[str, object],
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
) -> None:
    """Utrwal pełny stan ukończonej epoki i oddziel najlepszy model od ostatniego."""

    checkpoint = {
        "format_version": 1,
        "architecture": config["architecture"],
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": state["history"]["val_loss"][-1],
        "rng_state": capture_rng_state(loaders, device),
        **state,
    }

    save_checkpoint(run_path / "last.pt", checkpoint)

    if state["best_epoch"] == state["epoch"]:
        save_checkpoint(run_path / "best.pt", checkpoint)

    save_history(state["history"], run_path)


def recover_run_artifacts(run_path: Path, checkpoint: dict[str, object]) -> None:
    """Odtwórz CSV i najlepszy zapis, jeżeli przerwano zapis ostatniej epoki."""

    if checkpoint["best_epoch"] == checkpoint["epoch"]:
        save_checkpoint(run_path / "best.pt", checkpoint)
    else:
        best = read_checkpoint(run_path / "best.pt")

        if (
            best["epoch"] != checkpoint["best_epoch"]
            or best["val_loss"] != checkpoint["best_val_loss"]
            or best["config"] != checkpoint["config"]
        ):
            raise ValueError("best.pt nie odpowiada stanowi uruchomienia w last.pt.")

    save_history(checkpoint["history"], run_path)


def restore_training(
    run_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    config: dict[str, object],
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
) -> dict[str, object]:
    """Odtwórz koniec epoki bez resetowania harmonogramu i najlepszego wyniku."""

    checkpoint = read_checkpoint(run_path / "last.pt")

    if checkpoint["config"] != config:
        raise ValueError(
            "Konfiguracja różni się od checkpointu; utwórz nowy eksperyment."
        )
    if config["environment"] != environment_info(device):
        raise ValueError("Wznowienie wymaga tego samego środowiska i typu urządzenia.")
    if any(
        loader.num_workers != config["num_workers"] or loader.persistent_workers
        for loader in loaders
    ):
        raise ValueError(
            "Wznowienie wymaga zapisanej liczby workerów i persistent_workers=False."
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    recover_run_artifacts(run_path, checkpoint)
    restore_rng_state(checkpoint["rng_state"], loaders, device)

    return {
        key: checkpoint[key]
        for key in ("epoch", "best_epoch", "best_val_loss", "history")
    }


def load_model(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, object]]:
    """Wczytaj wagi do zapisanej architektury bez potrzeby ładowania datasetu."""

    checkpoint = read_checkpoint(path)
    architecture = checkpoint["architecture"]
    model = build_model(architecture["name"], architecture["params"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint
