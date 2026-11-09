"""Uczenie modeli kolejno na wspólnym zapisanym podziale danych."""

from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .checkpoints import (
    create_run,
    environment_info,
    initial_training_state,
    load_run_config,
    record_epoch,
    restore_training,
    save_epoch,
    seed_everything,
)
from .data import load_partitions, make_training_loaders, read_split_info
from .models import build_model
from .paths import display_path


DEFAULT_TRAINING = {
    "scale_factor": 2,
    "hr_size": 256,
    "batch_size": 64,
    "eval_batch_size": 64,
    "learning_rate": 0.005,
    "lr_step_size": 10,
    "lr_gamma": 0.75,
    "train_limit": 2560,
    "val_limit": 1280,
    "augment_train": True,
    "seed": 42,
    "num_workers": 0,
}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    """Przejdź raz przez trening i zwróć średnią stratę na przetworzony obraz."""

    model.train()
    total_loss = torch.zeros((), device=device)
    image_count = 0

    for lr, hr in tqdm(
        loader,
        desc="Trening",
        unit="partia",
        leave=False,
        mininterval=0.5,
        disable=not show_progress,
    ):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        sr = model(lr)
        loss = criterion(sr, hr)

        loss.backward()
        optimizer.step()

        batch_size = lr.shape[0]
        total_loss += loss.detach() * batch_size
        image_count += batch_size

    if image_count == 0:
        raise ValueError("Brak obrazów do treningu.")

    return (total_loss / image_count).item()


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    show_progress: bool = True,
) -> float:
    """Oceń bieżące wagi na danych walidacyjnych i zwróć średnią stratę na obraz."""

    model.eval()
    total_loss = torch.zeros((), device=device)
    image_count = 0

    for lr, hr in tqdm(
        loader,
        desc="Walidacja",
        unit="partia",
        leave=False,
        mininterval=0.5,
        disable=not show_progress,
    ):
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        sr = model(lr)
        loss = criterion(sr, hr)

        batch_size = lr.shape[0]
        total_loss += loss * batch_size
        image_count += batch_size

    if image_count == 0:
        raise ValueError("Brak obrazów do walidacji.")

    return (total_loss / image_count).item()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    device: torch.device,
    epochs: int,
    run_path: Path,
    config: dict[str, object],
    resume: bool = False,
    show_progress: bool = True,
) -> dict[str, list[float]]:
    """Ucz do łącznego celu epok, zapisując każdy ukończony krok eksperymentu."""

    if epochs <= 0:
        raise ValueError("Liczba epok musi być dodatnia.")
    if config != load_run_config(run_path):
        raise ValueError("Konfiguracja została zmieniona; utwórz nowy eksperyment.")

    loaders = (train_loader, val_loader)

    if resume:
        state = restore_training(
            run_path, model, optimizer, scheduler, config, loaders, device
        )
    else:
        if any(
            (run_path / name).exists() for name in ("last.pt", "best.pt", "history.csv")
        ):
            raise FileExistsError(
                "Katalog ma już trening; wybierz resume lub nowy run."
            )
        state = initial_training_state()

    if state["epoch"] >= epochs:
        print(
            f"Osiągnięto już {state['epoch']} epok; cel {epochs} nie wymaga treningu."
        )
        return state["history"]

    for epoch in range(state["epoch"] + 1, epochs + 1):
        started = perf_counter()
        learning_rate = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, show_progress
        )
        val_loss = validate(model, val_loader, criterion, device, show_progress)

        scheduler.step()
        duration = perf_counter() - started
        record_epoch(state, train_loss, val_loss, learning_rate, duration)
        save_epoch(
            run_path, model, optimizer, scheduler, config, state, loaders, device
        )

        print(
            f"Epoka {epoch}/{epochs} · {config['loss_name']}: "
            f"train={train_loss:.6f}, val={val_loss:.6f}, "
            f"lr={learning_rate:.1e} | {duration:.1f} s",
            flush=True,
        )

    return state["history"]


def build_training_components(
    config: dict[str, object], device: torch.device
) -> tuple[
    nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.StepLR
]:
    """Przygotuj model, MSE, Adama i StepLR według jednego configu."""

    architecture = config["architecture"]
    model = build_model(architecture["name"], architecture["params"]).to(device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"]
    )

    return model, criterion, optimizer, scheduler


def validate_model_specs(models: list[dict[str, object]]) -> None:
    """Sprawdź krótką listę modeli, nie dopuszczając ustawień treningu per model."""

    if not models:
        raise ValueError("Dodaj co najmniej jeden model do MODELS.")

    names = []

    for model in models:
        if set(model) != {"name", "architecture", "params"}:
            raise ValueError("Model wymaga wyłącznie pól name, architecture i params.")

        name = model["name"]

        if not name or any(not (char.isalnum() or char in "_-") for char in name):
            raise ValueError("Nazwa eksperymentu może zawierać litery, cyfry, _ i -.")
        if "scale_factor" in model["params"]:
            raise ValueError("Skala jest wspólnym ustawieniem TRAINING.")

        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("Nazwy modeli w kolejce muszą być różne.")


def make_run_config(
    model: dict[str, object],
    training: dict[str, object],
    dataset_path: Path,
    epochs: int,
    device: torch.device,
) -> dict[str, object]:
    """Rozwiń wspólne ustawienia i zapisz tożsamość gotowego podziału danych."""

    unknown = set(training) - set(DEFAULT_TRAINING)

    if unknown:
        raise ValueError(f"Nieznane ustawienia treningu: {sorted(unknown)}")

    settings = {**DEFAULT_TRAINING, **training}
    scale = settings.pop("scale_factor")

    if epochs <= 0:
        raise ValueError("Wybierz dodatnią liczbę epok.")
    if not isinstance(settings["num_workers"], int) or settings["num_workers"] < 0:
        raise ValueError("num_workers musi być całkowitą liczbą nieujemną.")

    return {
        **settings,
        "format_version": 1,
        "experiment_name": model["name"],
        "architecture": {
            "name": model["architecture"],
            "params": {**model["params"], "scale_factor": scale},
        },
        "loss_name": "MSE",
        "loss_reduction": "mean",
        "loss_clamp_prediction": False,
        "optimizer": "Adam",
        "scheduler": "StepLR",
        "initial_target_epochs": epochs,
        "dataset_path": display_path(dataset_path),
        "partition_info": read_split_info(dataset_path),
        "environment": environment_info(device),
        "metrics": {
            "color_space": "RGB",
            "data_range": 1.0,
            "clamp_prediction": True,
            "border_crop": 0,
            "psnr_aggregation": "mean_per_image_db",
            "ssim_gaussian_weights": True,
            "ssim_sigma": 1.5,
            "ssim_use_sample_covariance": False,
        },
    }


def train_experiment(
    config: dict[str, object],
    dataset_path: Path,
    experiments_path: Path,
    epochs: int,
    device: torch.device,
    show_progress: bool = True,
) -> Path:
    """Wytrenuj jeden model; po powrocie jego obiekty GPU nie są przechowywane."""

    seed_everything(config["seed"], device)
    parts = load_partitions(dataset_path)
    train_loader, val_loader = make_training_loaders(parts, config)
    model, criterion, optimizer, scheduler = build_training_components(config, device)
    run_path = create_run(experiments_path, config)
    print(f"Eksperyment: {display_path(run_path)}", flush=True)
    print(
        f"Uczenie: {len(train_loader.dataset)}; "
        f"walidacja robocza: {len(val_loader.dataset)}"
    )
    fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        run_path,
        config,
        show_progress=show_progress,
    )

    return run_path


def train_models(
    models: list[dict[str, object]],
    training: dict[str, object],
    dataset_path: Path,
    experiments_path: Path,
    epochs: int,
    device: torch.device,
    show_progress: bool = True,
) -> list[Path]:
    """Wykonaj kolejkę modeli z identycznymi ustawieniami i świeżym losowaniem."""

    validate_model_specs(models)
    seed_everything(training.get("seed", 42), device)
    configs = [
        make_run_config(item, training, dataset_path, epochs, device) for item in models
    ]
    paths = []

    for index, config in enumerate(configs, start=1):
        print(f"Model {index}/{len(configs)}: {config['experiment_name']}", flush=True)
        path = train_experiment(
            config, dataset_path, experiments_path, epochs, device, show_progress
        )
        paths.append(path)

    return paths


def resume_experiment(
    run_path: Path,
    dataset_path: Path,
    epochs: int,
    device: torch.device,
    show_progress: bool = True,
) -> Path:
    """Kontynuuj ostatnią ukończoną epokę według zapisanej konfiguracji."""

    config = load_run_config(run_path)

    if config["partition_info"] != read_split_info(dataset_path):
        raise ValueError("Wybrany podział danych różni się od eksperymentu.")

    seed_everything(config["seed"], device)
    parts = load_partitions(dataset_path)
    train_loader, val_loader = make_training_loaders(parts, config)
    model, criterion, optimizer, scheduler = build_training_components(config, device)
    fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        run_path,
        config,
        resume=True,
        show_progress=show_progress,
    )

    return run_path


def resume_models(
    run_paths: list[Path],
    dataset_path: Path,
    epochs: int,
    device: torch.device,
    show_progress: bool = True,
) -> list[Path]:
    """Wznów wybrane katalogi kolejno do wspólnego docelowego numeru epoki."""

    if not run_paths:
        raise ValueError("Dodaj katalogi eksperymentów do RESUME_PATHS.")

    return [
        resume_experiment(path, dataset_path, epochs, device, show_progress)
        for path in run_paths
    ]
