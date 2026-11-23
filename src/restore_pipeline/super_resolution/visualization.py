"""Czytelne wykresy i te same fragmenty obrazów dla porównań modeli."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .inference import upscale_bicubic

BLUE = "#1F5A85"
GRAY = "#8B9299"
TEXT = "#374151"
GRID = "#E5E7EB"


def set_plot_style() -> None:
    """Ustaw wspólny styl wykresów bez zmiany wyglądu przy samym imporcie."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "font.size": 10,
            "axes.titlelocation": "left",
            "axes.titlesize": 12,
            "figure.dpi": 110,
        }
    )


def style_axis(axis: Axes, grid_axis: str = "y") -> None:
    """Usuń zbędne obramowania i pozostaw delikatną siatkę wspierającą odczyt."""

    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)

    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(length=0, pad=8)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)


def epoch_ticks(first: int, last: int) -> list[int]:
    """Ogranicz etykiety osi do istniejących epok i uwzględnij oba końce."""

    locator = MaxNLocator(nbins=5, integer=True)
    ticks = [
        int(value)
        for value in locator.tick_values(first, max(first + 1, last))
        if first <= value <= last
    ]

    if first not in ticks:
        ticks.insert(0, first)
    if last not in ticks:
        ticks.append(last)

    return ticks


def plot_loss_panel(
    axis: Axes,
    history: dict[str, list[float]],
    start_epoch: int,
    title: str,
) -> None:
    """Pokaż wybrany zakres epok z etykietami bezpośrednio przy liniach."""

    epochs = np.arange(start_epoch, len(history["train_loss"]) + 1)
    marker = "o" if len(epochs) == 1 else None

    train_offset = -12 if history["train_loss"][-1] <= history["val_loss"][-1] else 12

    for key, label, color, offset in (
        ("train_loss", "Trening", GRAY, train_offset),
        ("val_loss", "Walidacja", BLUE, -train_offset),
    ):
        values = history[key][start_epoch - 1 :]
        axis.plot(epochs, values, color=color, linewidth=2, marker=marker)
        axis.annotate(
            label,
            (epochs[-1], values[-1]),
            xytext=(8, offset),
            textcoords="offset points",
            va="center",
            color=color,
        )

    best = int(np.argmin(history["val_loss"])) + 1
    if best >= start_epoch:
        axis.scatter(best, history["val_loss"][best - 1], color=BLUE, s=35, zorder=3)

    margin = max((epochs[-1] - epochs[0]) * 0.2, 1.0)
    axis.set(
        title=title,
        xlabel="Epoka",
        ylabel="Strata · MSE",
        xlim=(epochs[0] - 0.3, epochs[-1] + margin),
    )
    axis.set_xticks(epoch_ticks(int(epochs[0]), int(epochs[-1])))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    style_axis(axis)


def plot_history(
    history: dict[str, list[float]],
    detail_start: int = 5,
) -> Figure:
    """Ujawnij późniejsze zmiany strat bez ukrywania początku treningu."""

    count = len(history["train_loss"])
    detail_start = max(1, min(detail_start, count))
    best = int(np.argmin(history["val_loss"])) + 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    plot_loss_panel(axes[0], history, 1, "Cały trening")
    plot_loss_panel(
        axes[1],
        history,
        detail_start,
        f"Zbliżenie · od epoki {detail_start}",
    )
    fig.suptitle(
        f"Najlepsza walidacja: {history['val_loss'][best - 1]:.5f} · epoka {best}",
        x=0.05,
        ha="left",
        color=TEXT,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def plot_learning_rate(history: dict[str, list[float]]) -> Figure:
    """Pokaż harmonogram kroku uczenia, jeśli zapisano go w historii."""

    epochs = np.arange(1, len(history["learning_rate"]) + 1)
    fig, axis = plt.subplots(figsize=(9, 2.6))
    axis.step(epochs, history["learning_rate"], where="post", color=BLUE, linewidth=2)
    axis.set(
        title="Learning rate użyty w kolejnych epokach",
        xlabel="Epoka",
        ylabel="Learning rate",
    )
    axis.set_xticks(epoch_ticks(1, len(epochs)))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
    style_axis(axis)
    fig.tight_layout()
    return fig


def plot_metric_panel(
    axis: Axes,
    results: dict[str, dict[str, float]],
    metric: str,
    title: str,
) -> None:
    """Pokaż model i bicubic na wspólnej skali jednej metryki."""

    values = [results[method][metric] for method in ("Model", "Bicubic")]
    finite = [value for value in values if np.isfinite(value)]
    low = min(0.0, min(finite, default=0.0))
    high = max(finite, default=1.0)
    span = max(high - low, 0.01)
    axis.set_xlim(low - 0.08 * span, high + 0.45 * span)

    for row, (value, color) in enumerate(zip(values, (BLUE, GRAY))):
        if np.isfinite(value):
            axis.scatter(value, row, color=color, s=65, zorder=3)
            text = f"{value:.5f}" if metric == "MSE" else f"{value:.3f}"
            axis.annotate(
                text,
                (value, row),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                color=color,
            )
        else:
            axis.text(
                0.95,
                row,
                "∞",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                color=color,
            )

    axis.set(
        title=title, ylim=(1.6, -0.6), yticks=[0, 1], yticklabels=["Model", "Bicubic"]
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    style_axis(axis, grid_axis="x")


def plot_metrics(
    results: dict[str, dict[str, float]],
    title: str = "Porównanie na pełnej walidacji",
) -> Figure:
    """Porównaj metryki o różnych jednostkach na osobnych osiach."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), sharey=True)
    panels = (
        ("MSE", "MSE · mniej = lepiej"),
        ("PSNR", "PSNR [dB] · więcej = lepiej"),
        ("SSIM", "SSIM · więcej = lepiej"),
    )

    for axis, (metric, label) in zip(axes, panels):
        plot_metric_panel(axis, results, metric, label)

    fig.suptitle(title, x=0.05, ha="left", color=TEXT, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def plot_gains(per_image: pd.DataFrame) -> Figure:
    """Pokaż, jak często i o ile model poprawia PSNR na pojedynczych obrazach."""

    gains = per_image["psnr_gain"].to_numpy()
    finite = gains[np.isfinite(gains)]
    win_rate = 100 * float((gains > 0).mean())
    fig, axis = plt.subplots(figsize=(9, 3.8))

    if finite.size:
        extent = max(float(np.abs(finite).max()), 0.05)
        counts, edges = np.histogram(finite, bins=np.linspace(-extent, extent, 31))
        axis.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=[GRAY if edge < 0 else BLUE for edge in edges[:-1]],
        )

    axis.axvline(0, color=TEXT, linewidth=1)
    axis.set(
        title=f"Model poprawia PSNR dla {win_rate:.1f}% obrazów",
        xlabel="Zmiana PSNR względem bicubic [dB] · dodatnia = poprawa",
        ylabel="Liczba obrazów",
    )
    if finite.size != gains.size:
        axis.text(
            0.98,
            0.95,
            f"Poza histogramem: {gains.size - finite.size} wartości ±∞",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=GRAY,
        )

    style_axis(axis)
    fig.tight_layout()
    return fig


@torch.inference_mode()
def comparison_images(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, Tensor]],
    index: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Zwróć LR, bicubic, SR i HR na CPU dla tego samego dzieła."""

    model.eval()
    lr, hr = dataset[index]
    sr = model(lr.unsqueeze(0).to(device, non_blocking=True)).cpu()[0].clamp(0, 1)
    bicubic = upscale_bicubic(lr.unsqueeze(0), (hr.shape[-1], hr.shape[-2]))[0]

    return lr, bicubic, sr, hr


def find_detail(image: Tensor, size: int) -> tuple[int, int]:
    """Znajdź wyrazisty fragment na prostej siatce bez korzystania z wyniku modelu."""

    height, width = image.shape[-2:]
    if not 2 <= size <= min(height, width):
        raise ValueError("Fragment musi mieć co najmniej 2 piksele i mieścić się w HR.")

    gray = TF.rgb_to_grayscale(image)[0]
    edges = (gray[1:, :-1] - gray[:-1, :-1]).abs()
    edges += (gray[:-1, 1:] - gray[:-1, :-1]).abs()
    best_score = -1.0
    best_position = (0, 0)

    for top in range(0, height - size + 1, max(size // 2, 1)):
        for left in range(0, width - size + 1, max(size // 2, 1)):
            score = edges[top : top + size - 1, left : left + size - 1].mean().item()
            if score > best_score:
                best_score = score
                best_position = (top, left)

    return best_position


def crop_comparison(
    images: tuple[Tensor, ...], size: int, mode: str
) -> tuple[Tensor, ...]:
    """Przygotuj porównywalne zbliżenia bez niezależnego wybierania fragmentów."""

    height, width = images[-1].shape[-2:]
    if not 2 <= size <= min(height, width):
        raise ValueError("Fragment musi mieć co najmniej 2 piksele i mieścić się w HR.")

    if mode == "edges":
        top, left = find_detail(images[-1], size)
    elif mode == "center":
        top, left = (height - size) // 2, (width - size) // 2
    else:
        raise ValueError("Wybierz detail_mode='center' albo 'edges'.")

    return tuple(image[:, top : top + size, left : left + size] for image in images[1:])


def show_predictions(
    model: nn.Module,
    dataset: Dataset[tuple[Tensor, Tensor]],
    indices: list[int],
    device: torch.device,
    detail_size: int | None = None,
    detail_mode: str = "edges",
    title: str | None = None,
) -> Figure:
    """Porównaj całe obrazy lub identyczne fragmenty bez zmiany skali kolorów."""

    if not indices:
        raise ValueError("Wybierz co najmniej jeden obraz.")

    titles = (
        ("LR", "Bicubic", "Model", "HR")
        if detail_size is None
        else ("Bicubic", "Model", "HR")
    )
    fig, axes = plt.subplots(
        len(indices), len(titles), figsize=(14, 3.2 * len(indices)), squeeze=False
    )

    for row, index in enumerate(indices):
        images = comparison_images(model, dataset, index, device)

        if detail_size is not None:
            images = crop_comparison(images, detail_size, detail_mode)

        for axis, image, column_title in zip(axes[row], images, titles):
            axis.imshow(
                image.permute(1, 2, 0).numpy(), interpolation="nearest", vmin=0, vmax=1
            )
            if row == 0:
                axis.set_title(
                    column_title, color=BLUE if column_title == "Model" else TEXT
                )
            axis.axis("off")

        axes[row, 0].text(
            -0.04,
            0.5,
            f"#{index}",
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            color=GRAY,
        )

    detail_label = "krawędzie" if detail_mode == "edges" else "środek"
    label = (
        "Pełne obrazy"
        if detail_size is None
        else f"Ten sam fragment {detail_size}×{detail_size} · {detail_label}"
    )
    fig.suptitle(title or label, x=0.05, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_benchmark_panel(
    axis: Axes,
    frame: pd.DataFrame,
    metric: str,
    title: str,
    higher: bool,
) -> None:
    """Pokaż wspólną metrykę z bezpośrednimi etykietami i wyróżnij najlepszy model."""

    values = frame[metric].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    low = min(float(finite.min()) if finite.size else 0.0, 0.0)
    high = max(float(finite.max()) if finite.size else 1.0, 0.01)
    span = high - low
    models = frame["method"].eq("Model").to_numpy()
    candidates = np.flatnonzero(models)
    best = candidates[
        np.argmax(values[candidates]) if higher else np.argmin(values[candidates])
    ]

    for row, value in enumerate(values):
        color = BLUE if row == best else GRAY

        if np.isfinite(value):
            axis.scatter(value, row, color=color, s=45, zorder=3)
            label = f"{value:.5f}" if metric == "MSE" else f"{value:.3f}"
            axis.annotate(
                label,
                (value, row),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=color,
            )
        else:
            axis.text(
                0.95,
                row,
                "∞",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                color=color,
            )

    axis.set(
        title=title,
        xlim=(low - 0.04 * span, high + 0.4 * span),
        ylim=(len(frame) - 0.5, -0.5),
        yticks=range(len(frame)),
        yticklabels=frame["label"].tolist(),
    )
    axis.xaxis.set_major_locator(MaxNLocator(nbins=3))
    style_axis(axis, "x")


def plot_benchmark(frame: pd.DataFrame) -> Figure:
    """Porównaj jakość wszystkich wybranych modeli i jednego punktu bicubic."""

    fig, axes = plt.subplots(1, 3, figsize=(14, max(3, 0.6 * len(frame) + 1)))

    for axis, metric, title, higher in zip(
        axes,
        ("MSE", "PSNR", "SSIM"),
        (
            "MSE · mniej = lepiej",
            "PSNR [dB] · więcej = lepiej",
            "SSIM · więcej = lepiej",
        ),
        (False, True, True),
        strict=True,
    ):
        plot_benchmark_panel(axis, frame, metric, title, higher)

    fig.suptitle("Porównanie jakości · pełny zbiór", x=0.05, ha="left", color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    return fig


def plot_inference_time(frame: pd.DataFrame) -> Figure:
    """Porównaj amortyzowany czas forward przy wspólnym batch size."""

    models = frame.loc[frame["method"].eq("Model")].reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(9, max(2.5, 0.6 * len(models) + 1)))
    plot_benchmark_panel(
        axis,
        models,
        "inference_ms_per_image",
        "Forward [ms/obraz] · mniej = lepiej",
        False,
    )
    axis.set_xlabel(
        f"Batch size: {int(models['batch_size'].iloc[0])}; bez transferów i metryk"
    )
    fig.tight_layout()

    return fig


def plot_user_prediction(image: Image.Image, result: Image.Image) -> Figure:
    """Zestaw zdjęcie użytkownika, bicubic i wynik modelu z wymiarami obrazów."""

    bicubic = image.resize(result.size, Image.Resampling.BICUBIC)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for axis, picture, label in zip(
        axes, (image, bicubic, result), ("Wejście", "Bicubic", "Model"), strict=True
    ):
        axis.imshow(picture, interpolation="nearest")
        axis.set_title(
            f"{label} · {picture.width}×{picture.height}",
            color=BLUE if label == "Model" else TEXT,
        )
        axis.axis("off")

    fig.tight_layout()

    return fig
