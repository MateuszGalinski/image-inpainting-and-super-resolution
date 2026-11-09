"""Architektury super resolution identyfikowane trwałą nazwą wersji."""

from torch import Tensor, nn


class ESPCN1(nn.Module):
    """Odtwarzaj RGB za pomocą szerokiej warstwy cech i PixelShuffle."""

    def __init__(self, scale_factor: int = 2, hidden_channels: int = 128) -> None:
        """Zbuduj dwie konwolucje i określ kontekst potrzebny przy podziale obrazu."""

        super().__init__()

        if scale_factor not in (2, 4) or hidden_channels <= 0:
            raise ValueError("Wybierz skalę 2 lub 4 i dodatnią liczbę kanałów.")

        self.scale_factor = scale_factor
        self.required_overlap = 3

        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 3 * scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, lr: Tensor) -> Tensor:
        """Zamień partię LR (N, 3, H, W) w SR (N, 3, skala·H, skala·W)."""

        return self.layers(lr)


class ESPCN3(nn.Module):
    """Zastosuj układ 64 → 32 i tanh z ESPCN Shi i in. (2016) do obrazów RGB."""

    def __init__(self, scale_factor: int = 2) -> None:
        """Zbuduj konwolucje 5×5, 3×3 i 3×3 oraz końcowy PixelShuffle."""

        super().__init__()

        if scale_factor not in (2, 4):
            raise ValueError("Wybierz skalę 2 lub 4.")

        self.scale_factor = scale_factor
        self.required_overlap = 4

        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(32, 3 * scale_factor**2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, lr: Tensor) -> Tensor:
        """Zamień partię LR (N, 3, H, W) w SR (N, 3, skala·H, skala·W)."""

        return self.layers(lr)


class ResidualBlock(nn.Module):
    """Ucz poprawkę cech, dodając wynik dwóch konwolucji do wejścia bloku."""

    def __init__(self, channels: int) -> None:
        """Zachowaj liczbę kanałów i rozmiar obrazu wewnątrz bloku."""

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Dodaj wyuczoną poprawkę do niezmienionych cech wejściowych."""

        return features + self.layers(features)


class EDSRSmall(nn.Module):
    """Odtwarzaj RGB małym wariantem EDSR z lokalnymi i globalnym skrótem."""

    def __init__(
        self,
        scale_factor: int = 2,
        hidden_channels: int = 16,
        num_blocks: int = 4,
    ) -> None:
        """Zbuduj bloki residual bez BatchNorm i powiększenie przez PixelShuffle."""

        super().__init__()

        if scale_factor not in (2, 4) or hidden_channels <= 0 or num_blocks <= 0:
            raise ValueError("Wybierz skalę 2 lub 4 oraz dodatnie kanały i bloki.")

        self.scale_factor = scale_factor
        self.required_overlap = 2 * num_blocks + 4

        self.head = nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(
            *[ResidualBlock(hidden_channels) for _ in range(num_blocks)],
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels * scale_factor**2,
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
        )

    def forward(self, lr: Tensor) -> Tensor:
        """Przetwórz cechy w LR, dodaj globalny skrót i odtwórz obraz SR w RGB."""

        features = self.head(lr)
        features = features + self.body(features)

        return self.tail(features)


def build_model(name: str, params: dict[str, int]) -> nn.Module:
    """Odtwórz wskazany wariant bez zgadywania architektury na podstawie wag."""

    if name == "espcn1_v1":
        return ESPCN1(**params)
    if name == "espcn3_v1":
        return ESPCN3(**params)
    if name == "edsr_small_v1":
        return EDSRSmall(**params)

    raise ValueError(f"Nieznana architektura: {name}.")
