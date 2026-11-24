from torch import Tensor

def _region_errors(
    restored: Tensor,
    clean: Tensor,
    mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    mask = mask.bool()

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("Expected masks of shape [B, 1, H, W].")

    error = (restored.float() - clean.float()).abs()
    channels = clean.shape[1]

    hole_pixels = mask.sum(dim=(1, 2, 3))
    valid_pixels = (~mask).sum(dim=(1, 2, 3))

    hole_l1 = (
        (error * mask).sum(dim=(1, 2, 3))
        / (hole_pixels * channels).clamp_min(1)
    )

    valid_l1 = (
        (error * (~mask)).sum(dim=(1, 2, 3))
        / (valid_pixels * channels).clamp_min(1)
    )

    return hole_l1, valid_l1, hole_pixels > 0


def hybrid_inpainting_loss(
    restored: Tensor,
    clean: Tensor,
    mask: Tensor,
) -> Tensor:
    hole_l1, valid_l1, has_hole = _region_errors(
        restored, clean, mask
    )

    masked_loss = hole_l1.sum() / has_hole.sum().clamp_min(1)

    return masked_loss + 0.1 * valid_l1.mean()