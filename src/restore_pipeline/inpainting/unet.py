import torch
import torch.nn as nn
from torch import Tensor


def double_convolution(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        num_conditions: int | None = None,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.num_conditions = num_conditions

        # Encoder
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_convolution_1 = double_convolution(3, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Decoder
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2,
        )

        self.up_convolution_1 = double_convolution(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2,
        )

        self.up_convolution_2 = double_convolution(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2,
        )

        self.up_convolution_3 = double_convolution(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2,
        )

        self.up_convolution_4 = double_convolution(128, 64)

        self.out = nn.Conv2d(
            in_channels=64,
            out_channels=num_classes,
            kernel_size=1,
        )

        # Optional conditioning
        if num_conditions is not None:
            self.condition_embedding = nn.Embedding(
                num_embeddings=num_conditions,
                embedding_dim=embedding_dim,
            )

            # FiLM produces gamma and beta for 1024 bottleneck channels
            self.condition_film = nn.Linear(
                in_features=embedding_dim,
                out_features=1024 * 2,
            )

            # Start with identity conditioning:
            # gamma = 0, beta = 0
            nn.init.zeros_(self.condition_film.weight)
            nn.init.zeros_(self.condition_film.bias)

    def forward(
        self,
        x: Tensor,
        condition_ids: Tensor | None = None,
    ) -> Tensor:

        # Encoder
        down_1 = self.down_convolution_1(x)

        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)

        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)

        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)

        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)

        # FiLM conditioning at the bottleneck
        if self.num_conditions is not None:
            if condition_ids is None:
                raise ValueError(
                    "condition_ids must be provided when conditioning is enabled."
                )

            condition_ids = condition_ids.to(
                device=x.device,
                dtype=torch.long,
            )

            embedding = self.condition_embedding(condition_ids)

            gamma, beta = self.condition_film(embedding).chunk(2, dim=1)

            gamma = gamma[:, :, None, None]
            beta = beta[:, :, None, None]

            down_9 = down_9 * (1 + gamma) + beta

        # Decoder
        up_1 = self.up_transpose_1(down_9)

        x = self.up_convolution_1(
            torch.cat([down_7, up_1], dim=1)
        )

        up_2 = self.up_transpose_2(x)

        x = self.up_convolution_2(
            torch.cat([down_5, up_2], dim=1)
        )

        up_3 = self.up_transpose_3(x)

        x = self.up_convolution_3(
            torch.cat([down_3, up_3], dim=1)
        )

        up_4 = self.up_transpose_4(x)

        x = self.up_convolution_4(
            torch.cat([down_1, up_4], dim=1)
        )

        return self.out(x)