import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import rgb_to_grayscale
from torch import nn
import numpy as np

def to_sobel(x: torch.Tensor) -> torch.Tensor:
    """
    RGB batch -> grayscale + Sobel edges, removing colour (as in DeepCluster).

    Args:
        x (torch.Tensor): RGB images, shape (B, 3, H, W).

    Returns:
        torch.Tensor: Shape (B, 3, H, W): gray, horizontal edges, vertical edges.
    """
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=x.device, dtype=x.dtype)
    
    kernels = torch.stack([sobel_x, sobel_x.T]).unsqueeze(1)   # (2, 1, 3, 3)
    
    gray = rgb_to_grayscale(x)
    edges = F.conv2d(gray, kernels, padding=1)
    return edges

@torch.inference_mode()
def gram_embedding(feature_map: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) conv features -> (B, C(C+1)/2) upper triangle of the Gram matrix."""
    batch, channels, height, width = feature_map.shape
    flat = feature_map.flatten(start_dim=2)                          # (B, C, H*W)
    gram = flat @ flat.transpose(1, 2) / (height * width)            # (B, C, C)
    rows, cols = torch.triu_indices(channels, channels)
    return gram[:, rows, cols]


@torch.inference_mode()
def encode_loader_gram(encoder_part: nn.Module, loader, device) -> np.ndarray:
    """Like encode_loader, but embeds each image as the Gram matrix of `encoder_part`'s output."""
    encoder_part.eval()
    feats = [gram_embedding(encoder_part(batch["image"].to(device))).cpu() for batch in loader]
    return torch.cat(feats).numpy()


if __name__ == "__main__":
    img = torch.rand(4, 3, 256, 256)
    print("input:", img.shape)
    gray = rgb_to_grayscale(img)
    print("gray:", gray.shape)
    sob = to_sobel(img)
    print("to_sobel:", sob.shape) 