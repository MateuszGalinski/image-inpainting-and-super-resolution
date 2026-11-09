"""Predykcja obrazu RGB i przetwarzanie dużych zdjęć fragmentami."""

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


def upscale_bicubic(lr: Tensor, target_size: tuple[int, int]) -> Tensor:
    """Powiększ partię CPU do rozmiaru (szerokość, wysokość), zachowując RGB."""

    return torch.stack(
        [
            TF.to_tensor(
                TF.to_pil_image(image).resize(
                    target_size,
                    Image.Resampling.BICUBIC,
                )
            )
            for image in lr
        ]
    )


@torch.inference_mode()
def predict_tile(
    model: nn.Module,
    image: Tensor,
    bounds: tuple[int, int, int, int],
    overlap: int,
    device: torch.device,
) -> Tensor:
    """Zwróć sam środek fragmentu po predykcji z kontekstem sąsiadów."""

    top, left, bottom, right = bounds
    padded_top = max(0, top - overlap)
    padded_left = max(0, left - overlap)
    padded_bottom = min(image.shape[-2], bottom + overlap)
    padded_right = min(image.shape[-1], right + overlap)

    patch = image[:, padded_top:padded_bottom, padded_left:padded_right]
    prediction = model(patch.unsqueeze(0).to(device, non_blocking=True))[0].cpu()
    scale = model.scale_factor
    offset_y = (top - padded_top) * scale
    offset_x = (left - padded_left) * scale

    return prediction[
        :,
        offset_y : offset_y + (bottom - top) * scale,
        offset_x : offset_x + (right - left) * scale,
    ]


@torch.inference_mode()
def predict_tiles(
    model: nn.Module,
    image: Tensor,
    device: torch.device,
    tile_size: int = 256,
    overlap: int | None = None,
    show_progress: bool = True,
) -> Tensor:
    """Ogranicz pamięć GPU, składając wynik na CPU bez mieszania sąsiednich pikseli."""

    overlap = model.required_overlap if overlap is None else overlap

    if tile_size <= 0 or overlap < model.required_overlap:
        raise ValueError(
            f"Rozmiar fragmentu musi być dodatni, a kontekst co najmniej "
            f"{model.required_overlap}."
        )

    model.eval()
    height, width = image.shape[-2:]
    scale = model.scale_factor
    result = torch.empty((3, height * scale, width * scale), dtype=image.dtype)
    positions = [
        (top, left)
        for top in range(0, height, tile_size)
        for left in range(0, width, tile_size)
    ]

    for top, left in tqdm(
        positions,
        desc="Powiększanie fragmentami",
        unit="fragment",
        leave=False,
        mininterval=0.5,
        disable=not show_progress,
    ):
        bottom = min(top + tile_size, height)
        right = min(left + tile_size, width)
        prediction = predict_tile(
            model, image, (top, left, bottom, right), overlap, device
        )
        result[:, top * scale : bottom * scale, left * scale : right * scale] = (
            prediction
        )

    return result


@torch.inference_mode()
def predict_image(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    tile_size: int | None = None,
    overlap: int | None = None,
    show_progress: bool = True,
) -> Image.Image:
    """Powiększ RGB, opcjonalnie zmniejszając zapotrzebowanie na pamięć GPU."""

    model.eval()
    lr = TF.to_tensor(image.convert("RGB"))

    if tile_size is None:
        sr = model(lr.unsqueeze(0).to(device, non_blocking=True))[0].cpu()
    else:
        sr = predict_tiles(model, lr, device, tile_size, overlap, show_progress)

    return TF.to_pil_image(sr.clamp(0, 1))
