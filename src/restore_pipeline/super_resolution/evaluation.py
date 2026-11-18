"""Metryki RGB i porównania modeli na walidacji lub teście."""

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor, nn
from torch.utils.data import DataLoader, SequentialSampler
from tqdm.auto import tqdm

from .checkpoints import load_model, read_checkpoint, save_csv
from .data import SuperResolutionDataset, load_partitions, make_loader, read_split_info
from .inference import upscale_bicubic
from .paths import display_path
from .visualization import (
    plot_benchmark,
    plot_gains,
    plot_inference_time,
    plot_metrics,
    set_plot_style,
    show_predictions,
)


def image_metrics(prediction: Tensor, target: Tensor) -> dict[str, float]:
    """Oceń jeden obraz CHW w RGB; identyczny obraz ma PSNR równy nieskończoności."""

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Oczekiwano dwóch obrazów CHW o identycznych wymiarach.")

    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Obrazy zawierają NaN lub nieskończoność.")

    predicted = (
        prediction.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
        .astype(np.float64)
    )
    expected = target.detach().cpu().permute(1, 2, 0).numpy().astype(np.float64)

    mse = float(np.mean((predicted - expected) ** 2))
    psnr = (
        float(peak_signal_noise_ratio(expected, predicted, data_range=1.0))
        if mse
        else float("inf")
    )

    ssim = structural_similarity(
        expected,
        predicted,
        data_range=1.0,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )

    return {"MSE": mse, "PSNR": psnr, "SSIM": float(ssim)}


def summarize_results(per_image: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Zbuduj zestawienie modelu i bicubic z tabeli wyników poszczególnych dzieł."""

    return {
        label: {
            metric.upper(): float(per_image[f"{prefix}_{metric}"].mean())
            for metric in ("mse", "psnr", "ssim")
        }
        for label, prefix in (("Model", "model"), ("Bicubic", "bicubic"))
    }


def check_evaluation_loader(loader: DataLoader) -> None:
    """Wymagaj pełnej i stałej kolejności, aby indeksy pasowały do nazw plików."""

    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Ocena wymaga shuffle=False i drop_last=False.")
    if len(loader.dataset) == 0:
        raise ValueError("Zbiór oceny jest pusty.")


def evaluate_bicubic(loader: DataLoader, show_progress: bool = True) -> pd.DataFrame:
    """Policz niezmienny punkt odniesienia raz dla całego porównania."""

    check_evaluation_loader(loader)
    rows = []

    for lr, hr in tqdm(
        loader,
        desc="Bicubic",
        unit="partia",
        leave=False,
        mininterval=0.5,
        disable=not show_progress,
    ):
        baseline = upscale_bicubic(lr, (hr.shape[-1], hr.shape[-2]))

        for prediction, target in zip(baseline, hr, strict=True):
            scores = image_metrics(prediction, target)
            rows.append(
                {
                    "index": len(rows),
                    **{
                        f"bicubic_{key.lower()}": value for key, value in scores.items()
                    },
                }
            )

    return pd.DataFrame(rows)


def synchronize(device: torch.device) -> None:
    """Poczekaj na GPU przed odczytaniem czasu rzeczywistego forward."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Oceń obrazy i zmierz sam forward po trzech rozgrzewkowych przebiegach."""

    check_evaluation_loader(loader)
    model.eval()
    rows = []
    total_seconds = 0.0

    for lr, hr in tqdm(
        loader,
        desc="Ocena modelu",
        unit="partia",
        leave=False,
        mininterval=0.5,
        disable=not show_progress,
    ):
        inputs = lr.to(device, non_blocking=True)

        if not rows:
            for _ in range(3):
                model(inputs)

        synchronize(device)
        started = perf_counter()
        prediction = model(inputs)
        synchronize(device)
        total_seconds += perf_counter() - started
        prediction = prediction.cpu()

        for image, target in zip(prediction, hr, strict=True):
            scores = image_metrics(image, target)
            rows.append(
                {
                    "index": len(rows),
                    **{f"model_{key.lower()}": value for key, value in scores.items()},
                }
            )

    timing = {
        "inference_ms_per_image": 1000 * total_seconds / len(rows),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "warmup_batches": 3,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
    }

    return pd.DataFrame(rows), timing


def combine_results(model_scores: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Połącz wyniki tych samych obrazów i wyznacz zyski względem bicubic."""

    if model_scores["index"].tolist() != baseline["index"].tolist():
        raise ValueError("Ocena modelu i bicubic dotyczy innych indeksów.")

    rows = model_scores.merge(baseline, on="index", validate="one_to_one")
    rows["mse_gain"] = rows["bicubic_mse"] - rows["model_mse"]
    rows["ssim_gain"] = rows["model_ssim"] - rows["bicubic_ssim"]
    rows["psnr_gain"] = 0.0
    different = rows["model_psnr"] != rows["bicubic_psnr"]
    rows.loc[different, "psnr_gain"] = (
        rows.loc[different, "model_psnr"] - rows.loc[different, "bicubic_psnr"]
    )

    return rows


def validate_experiments(
    run_paths: list[Path], partition_info: dict[str, object]
) -> dict[str, object]:
    """Sprawdź porównywalność wszystkich eksperymentów przed rozpoczęciem oceny."""

    if not run_paths or len({path.resolve() for path in run_paths}) != len(run_paths):
        raise ValueError("Wybierz niepustą listę różnych katalogów eksperymentów.")

    reference = None

    for path in run_paths:
        config = read_checkpoint(path / "best.pt")["config"]
        signature = {
            "partition_info": config["partition_info"],
            "hr_size": config["hr_size"],
            "scale_factor": config["architecture"]["params"]["scale_factor"],
            "metrics": config["metrics"],
        }

        if signature["partition_info"] != partition_info:
            raise ValueError(f"Inny podział danych w eksperymencie {path.name}.")
        if reference is not None and signature != reference:
            raise ValueError(
                "Porównanie wymaga tego samego podziału, HR, skali i metryk."
            )

        reference = signature

    return reference


def save_figure(figure: plt.Figure, path: Path) -> None:
    """Zapisz gotowy wykres i zwolnij jego pamięć po eksporcie."""

    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_report_figures(
    model: nn.Module,
    loader: DataLoader,
    per_image: pd.DataFrame,
    results: dict[str, dict[str, float]],
    output_path: Path,
    device: torch.device,
    preview_indices: list[int],
    case_count: int,
) -> None:
    """Zapisz stałe przykłady oraz mocne i słabe przypadki modelu."""

    save_figure(plot_metrics(results), output_path / "metrics.png")
    save_figure(plot_gains(per_image), output_path / "gains.png")
    size = min(64, loader.dataset.hr_size)
    best = per_image.nlargest(case_count, "psnr_gain")["index"].astype(int).tolist()
    worst = per_image.nsmallest(case_count, "psnr_gain")["index"].astype(int).tolist()

    for filename, indices, detail, title in (
        ("previews.png", preview_indices, None, "Stałe przykłady"),
        ("details.png", preview_indices, size, "Te same fragmenty · krawędzie"),
        ("best_cases.png", best, size, "Najlepsze przypadki względem bicubic"),
        ("worst_cases.png", worst, size, "Najsłabsze przypadki względem bicubic"),
    ):
        figure = show_predictions(
            model, loader.dataset, indices, device, detail_size=detail, title=title
        )
        save_figure(figure, output_path / filename)


def evaluate_experiment(
    run_path: Path,
    loader: DataLoader,
    baseline: pd.DataFrame,
    output_name: str,
    device: torch.device,
    preview_indices: list[int],
    case_count: int,
    show_progress: bool,
) -> dict[str, object]:
    """Oceń jeden najlepszy model i zapisz raport bez zmieniania checkpointów."""

    model, checkpoint = load_model(run_path / "best.pt", device)
    scores, timing = evaluate_model(model, loader, device, show_progress)
    per_image = combine_results(scores, baseline)
    per_image["checkpoint_epoch"] = checkpoint["epoch"]
    results = summarize_results(per_image)
    output_path = run_path / output_name
    output_path.mkdir(exist_ok=True)
    save_csv(per_image, output_path / "per_image.csv")

    summary = pd.DataFrame.from_dict(results, orient="index").rename_axis("method")
    summary = summary.reset_index()
    summary["checkpoint_epoch"] = checkpoint["epoch"]
    summary["loss_name"] = "MSE"
    summary["architecture"] = checkpoint["architecture"]["name"]
    for key, value in timing.items():
        summary[key] = value
    summary.loc[summary["method"].eq("Bicubic"), "inference_ms_per_image"] = np.nan
    summary.loc[summary["method"].eq("Bicubic"), "parameters"] = 0
    save_csv(summary, output_path / "summary.csv")
    save_report_figures(
        model,
        loader,
        per_image,
        results,
        output_path,
        device,
        preview_indices,
        case_count,
    )

    return {
        "method": "Model",
        "name": checkpoint["config"]["experiment_name"],
        "run_path": display_path(run_path),
        "checkpoint_epoch": checkpoint["epoch"],
        **results["Model"],
        **timing,
    }


def save_comparison(
    rows: list[dict[str, object]],
    baseline: pd.DataFrame,
    results_path: Path,
    output_name: str,
) -> pd.DataFrame:
    """Dodaj jeden punkt bicubic i zapisz zbiorcze wyniki wybranych eksperymentów."""

    baseline_row = {
        "method": "Bicubic",
        "name": "Bicubic",
        "parameters": 0,
        **{
            key.upper(): float(baseline[f"bicubic_{key}"].mean())
            for key in ("mse", "psnr", "ssim")
        },
    }
    frame = pd.DataFrame([*rows, baseline_row])
    duplicate = frame["name"].duplicated(keep=False)
    frame["label"] = [
        f"{name} ({index + 1})" if duplicate.iloc[index] else name
        for index, name in enumerate(frame["name"])
    ]
    results_path.mkdir(parents=True, exist_ok=True)
    save_csv(frame, results_path / f"{output_name}_comparison.csv")
    save_figure(plot_benchmark(frame), results_path / f"{output_name}_comparison.png")
    save_figure(
        plot_inference_time(frame), results_path / f"{output_name}_inference_time.png"
    )

    return frame


def benchmark_experiments(
    run_paths: list[Path],
    dataset_path: Path,
    results_path: Path,
    device: torch.device,
    split: str = "val",
    batch_size: int = 64,
    preview_indices: list[int] | None = None,
    case_count: int = 3,
    show_progress: bool = True,
    num_workers: int = 0,
) -> pd.DataFrame:
    """Porównaj pełną walidację albo test, wyliczając bicubic tylko raz."""

    if split not in ("val", "test") or case_count < 1:
        raise ValueError("Wybierz val albo test oraz dodatnią liczbę przypadków.")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers musi być całkowitą liczbą nieujemną.")

    reference = validate_experiments(run_paths, read_split_info(dataset_path))
    parts = load_partitions(dataset_path)
    source = parts[split]
    indices = [5, 6, 7, 8, 9] if preview_indices is None else preview_indices

    if not indices or any(index < 0 or index >= len(source) for index in indices):
        raise ValueError("Indeksy podglądu muszą należeć do ocenianego zbioru.")

    dataset = SuperResolutionDataset(
        source, reference["hr_size"], reference["scale_factor"]
    )
    loader = make_loader(
        dataset,
        batch_size,
        False,
        seed=42,
        num_workers=num_workers,
        persistent_workers=True,
    )
    baseline = evaluate_bicubic(loader, show_progress)
    baseline["filename"] = list(source["filename"])
    output_name = "validation" if split == "val" else "test"
    rows = []
    set_plot_style()

    for index, path in enumerate(run_paths, start=1):
        print(f"Ocena {index}/{len(run_paths)}: {path.name}", flush=True)
        rows.append(
            evaluate_experiment(
                path,
                loader,
                baseline,
                output_name,
                device,
                indices,
                case_count,
                show_progress,
            )
        )

    return save_comparison(rows, baseline, results_path, output_name)
