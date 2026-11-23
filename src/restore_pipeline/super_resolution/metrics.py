"""Metryki RGB dla benchmarku i pojedynczych zdjęć."""

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import Tensor


def image_metrics(
    prediction: Tensor, target: Tensor, *, calculate_ssim: bool = True
) -> dict[str, float]:
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

    ssim = float("nan")
    if calculate_ssim:
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
