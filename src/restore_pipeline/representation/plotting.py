import matplotlib.pyplot as plt
import torch
import torch.nn as nn


@torch.inference_mode()
def plot_reconstructions(
    model: nn.Module,
    images: list[torch.Tensor],
    device: str,
    titles: list[str] | None = None,
) -> None:
    """
    Show each image next to the model's output for it: one row per image.

    Args:
        model (nn.Module): Model whose forward returns an image of the same shape.
        images (list[torch.Tensor]): Images of shape (3, H, W), values in [0, 1].
        device (str): Device the model is on.
        titles (list[str] | None): Row labels, e.g. ["clean", "damaged"].
            Defaults to the image's position in the list.
    """
    model.eval()
    outputs = model(torch.stack(images).to(device)).cpu()
    titles = titles or [str(i) for i in range(len(images))]

    fig, axes = plt.subplots(len(images), 2, figsize=(8, 4 * len(images)), squeeze=False)
    for row, (before, after, title) in enumerate(zip(images, outputs, titles)):
        for col, (img, label) in enumerate([(before, "input"), (after, "output")]):
            axes[row, col].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            axes[row, col].set_title(f"{title}: {label}")
            axes[row, col].axis("off")
    plt.tight_layout()
    plt.show()
