import hashlib
import math
from typing import Callable

import torch

DamageFn = Callable[[torch.Tensor, torch.Generator], torch.Tensor]


def random_ellipse_damage(
    image: torch.Tensor,
    generator: torch.Generator,
    max_area_frac: float = 1 / 16,
    fill: float = 0.0,
) -> torch.Tensor:
    """
    Paint one randomly sized, randomly placed ellipse onto a (C, H, W) image.

    The ellipse area is drawn uniformly up to `max_area_frac` of the image area
    (clipping at the borders can only make it smaller). Aspect ratio is in
    [0.5, 2]. Returns a damaged copy; the input is not modified.
    """
    _, height, width = image.shape

    area = torch.rand(1, generator=generator).item() * max_area_frac * height * width
    aspect = 0.5 + 1.5 * torch.rand(1, generator=generator).item()   # semi-axis ratio a / b
    semi_x = math.sqrt(area * aspect / math.pi)
    semi_y = semi_x / aspect

    center_x = torch.rand(1, generator=generator).item() * width
    center_y = torch.rand(1, generator=generator).item() * height

    ys = torch.arange(height).view(-1, 1)
    xs = torch.arange(width).view(1, -1)
    inside = ((xs - center_x) / max(semi_x, 1e-6)) ** 2 + ((ys - center_y) / max(semi_y, 1e-6)) ** 2 <= 1

    damaged = image.clone()
    damaged[:, inside] = fill
    return damaged


def _random_generator() -> torch.Generator:
    """Generator seeded from OS entropy. (A bare `torch.Generator()` has a FIXED seed.)"""
    generator = torch.Generator()
    generator.seed()
    return generator


def _content_generator(image: torch.Tensor) -> torch.Generator:
    """Generator seeded from the pixels, so an image always gets the same damage."""
    digest = hashlib.sha256(image.numpy().tobytes()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


class PairTransform:
    """
    `with_transform` callable that turns a batch of raw images into
    {'image': input, 'target': clean} tensors.

    A class instead of a closure so it can be pickled for DataLoader workers
    (required on Windows).

    Args:
        base_transform: Converts a raw image to a (C, H, W) float tensor.
        damage_fn: Damage function. Ignored when `damage` is False.
        damage: If False, input equals target (a clean pair).
        deterministic: If True, damage is seeded from image content so it is
            identical on every read (use for validation and test). If False,
            damage is fresh random each time (use for training).
    """

    def __init__(self, base_transform, damage_fn: DamageFn = random_ellipse_damage,
                 damage: bool = True, deterministic: bool = False):
        self.base_transform = base_transform
        self.damage_fn = damage_fn
        self.damage = damage
        self.deterministic = deterministic

    def __call__(self, batch: dict) -> dict:
        clean = [self.base_transform(img) for img in batch["image"]]
        if not self.damage:
            return {"image": clean, "target": clean}

        damaged = [
            self.damage_fn(x, _content_generator(x) if self.deterministic else _random_generator())
            for x in clean
        ]
        return {"image": damaged, "target": clean}
