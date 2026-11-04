import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, img_size = 256, latent_dim = 128) -> None:
        super().__init__()
        final_channels_count = 128
        final_img_size = img_size // 16
        flatten_in_size = final_channels_count * final_img_size * final_img_size 

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2,2),
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
            nn.ConvTranspose2d(16, 3, kernel_size=3, padding=1, output_padding=1, stride=2), nn.Sigmoid(),
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x