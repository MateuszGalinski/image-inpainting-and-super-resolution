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


# Class name -> class. Checkpoints store the name; load_checkpoint looks it up here.
# Register every new model, one entry per class (sizes/layers go in init kwargs).
MODEL_REGISTRY: dict[str, type[RepresentationModel]] = {
    "ConvAutoencoder": ConvAutoencoder,
    "ConvOnlyAutoencoder": ConvOnlyAutoencoder,
}
