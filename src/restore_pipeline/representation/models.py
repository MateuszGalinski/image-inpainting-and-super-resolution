from typing import Any

import torch
import torch.nn as nn
from restore_pipeline.representation.utils import to_sobel

@torch.inference_mode()
def encode_loader(model, loader, device):
    """
    Each representation model can be used for this function
    """
    model.eval()
    feats = []
    for batch in loader:
        z = model.encode(batch["image"].to(device, non_blocking=True))
        feats.append(z.cpu())
    return torch.cat(feats).numpy()


class RepresentationModel(nn.Module):
    """
    Base class for models that can embed images.

    Stores the constructor kwargs in `init_kwargs` so a checkpoint can rebuild
    the exact architecture. Kwargs must be plain data (numbers, strings, bools,
    lists/tuples/dicts of those), because checkpoints are loaded with
    `weights_only=True`.
    """

    def __init__(self, **init_kwargs) -> None:
        super().__init__()
        self.init_kwargs = init_kwargs

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the embedding of a batch of images, shape (B, D)."""
        raise NotImplementedError


class ConvAutoencoder(RepresentationModel):
    def __init__(self, img_size = 256, in_channels = 3, out_channels = 3, latent_dim = 128) -> None:
        super().__init__(img_size=img_size, latent_dim=latent_dim)
        final_channels_count = 128
        final_img_size = img_size // 16
        flatten_in_size = final_channels_count * final_img_size * final_img_size 

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, final_channels_count, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Flatten(),
            nn.Linear(flatten_in_size, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flatten_in_size), nn.ReLU(inplace=True),
            nn.Unflatten(1, (final_channels_count, final_img_size, final_img_size)),
            nn.ConvTranspose2d(final_channels_count, 64, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=1, output_padding=1, stride=2), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

class ConvOnlyAutoencoder(RepresentationModel):
    def __init__(self, img_size = 256, in_channels = 3, out_channels = 3, latent_channels = 8) -> None:
        super().__init__(img_size=img_size, final_channels_count = latent_channels)
        final_channels_count = latent_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
            nn.Conv2d(64, final_channels_count, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(final_channels_count, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=1, output_padding=1, stride=2), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encode_non_flat(x).flatten(start_dim=1)

    def encode_non_flat(self, x) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        x = self.encode_non_flat(x)
        x = self.decode(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # 1. Residual branch F(x): Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False), 
            nn.BatchNorm2d(out_channels), 
        )
        self.act = nn.ReLU(inplace=True)

        if (stride == 1) and (in_channels == out_channels):
          self.shortcut = nn.Identity()
        else:
          self.shortcut = nn.Sequential(
              nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # we have batchnorm so no bias
              nn.BatchNorm2d(out_channels)
          )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.body(x)
        x = self.shortcut(x)
        return self.act(residual + x)

class DeepConvOnlyAutoencoder(RepresentationModel):
    """
    Deeper ConvOnly autoencoder with residual blocks in both encoder and decoder.

    Encoder: 4 stages of conv -> ReLU -> MaxPool -> `blocks_per_stage` ResidualBlocks,
    then a 1x1 conv to `latent_channels`. For 256 x 256 input the latent is
    (latent_channels, 16, 16); the default 32 x 16 x 16 = 8192 values matches
    ConvOnlyAutoencoder's latent size (24x compression for RGB), so comparisons
    change depth, not compression.

    Decoder mirrors it: at every resolution `blocks_per_stage` ResidualBlocks, then
    a ConvTranspose2d that doubles the resolution. kernel_size=4, stride=2,
    padding=1 gives exactly 2x and, because the kernel divides evenly by the
    stride, fewer checkerboard artifacts than kernel_size=3.
    """

    def __init__(self, img_size = 256, in_channels = 3, out_channels = 3,
                 latent_channels = 32, blocks_per_stage = 2) -> None:
        super().__init__(img_size=img_size, in_channels=in_channels, out_channels=out_channels,
                         latent_channels=latent_channels, blocks_per_stage=blocks_per_stage)
        channels = [
            (in_channels, 32),
            (32, 64),
            (64, 128),
            (128, 256),
        ]

        def res_blocks(c: int) -> list[nn.Module]:
            return [ResidualBlock(c, c) for _ in range(blocks_per_stage)]

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16
        encoder_layers = []
        for in_c, out_c in channels:
            encoder_layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
                *res_blocks(out_c),
            ])
        encoder_layers.append(nn.Conv2d(channels[-1][1], latent_channels, kernel_size=1))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder: 16 -> 32 -> 64 -> 128 -> 256
        decoder_layers = [
            nn.Conv2d(latent_channels, channels[-1][1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        reversed_channels = list(reversed(channels))
        for index, (in_c, out_c) in enumerate(reversed_channels):
            is_output_layer = index == len(reversed_channels) - 1
            decoder_layers.extend(res_blocks(out_c))
            decoder_layers.append(
                nn.ConvTranspose2d(
                    out_c,
                    out_channels if is_output_layer else in_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
            decoder_layers.append(nn.Sigmoid() if is_output_layer else nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encode_non_flat(x).flatten(start_dim=1)

    def encode_non_flat(self, x) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encode_non_flat(x)
        x = self.decode(x)
        return x


# Class name -> class. Checkpoints store the name; load_checkpoint looks it up here.
# Register every new model, one entry per class (sizes/layers go in init kwargs).
MODEL_REGISTRY: dict[str, type[RepresentationModel]] = {
    "ConvAutoencoder": ConvAutoencoder,
    "ConvOnlyAutoencoder": ConvOnlyAutoencoder,
    "DeepConvOnlyAutoencoder": DeepConvOnlyAutoencoder,
}
