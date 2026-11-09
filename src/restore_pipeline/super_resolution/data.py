"""Przygotowanie WikiArt i tworzenie par LR/HR."""

import json
from pathlib import Path

import torch
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .paths import display_path


def load_wikiart(path: Path) -> HFDataset:
    """Wczytaj lokalną bazę po preprocessingu bez ponownego pobierania."""

    if not (path / "dataset_info.json").is_file():
        raise FileNotFoundError(f"Brak przygotowanego zbioru: {display_path(path)}")

    dataset = load_from_disk(str(path))

    if not isinstance(dataset, HFDataset) or "image" not in dataset.column_names:
        raise ValueError("Oczekiwano pojedynczego Dataset z kolumną image.")

    return dataset


def split_wikiart(
    dataset: HFDataset,
    seed: int,
) -> tuple[HFDataset, HFDataset, HFDataset]:
    """Odtwórz podział matg; zgodność wymaga identycznej bazy i kolejności."""

    split = dataset.train_test_split(test_size=0.2, seed=seed)
    holdout = split["test"].train_test_split(test_size=0.5, seed=seed)

    return split["train"], holdout["train"], holdout["test"]


class SuperResolutionDataset(Dataset[tuple[Tensor, Tensor]]):
    """Twórz pomniejszone wejście LR i referencję HR z tego samego obrazu."""

    def __init__(
        self,
        images: HFDataset,
        hr_size: int,
        scale_factor: int,
        augment: bool = False,
    ) -> None:
        """Zachowaj źródło i rozmiary par bez kopiowania obrazów."""

        if scale_factor <= 1 or hr_size <= 0 or hr_size % scale_factor != 0:
            raise ValueError("Niepoprawny rozmiar HR lub współczynnik skali.")

        self.images = images.select_columns(["image"])
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.augment = augment

    def __len__(self) -> int:
        """Zwróć liczbę dostępnych par."""

        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Zdekoduj obraz i pomniejsz go bicubic, zachowując oryginał jako HR."""

        hr_image = self.images[index]["image"].convert("RGB")

        if hr_image.size != (self.hr_size, self.hr_size):
            raise ValueError(
                f"Obraz {index}: rozmiar {hr_image.size}, "
                f"oczekiwano {self.hr_size}×{self.hr_size}."
            )

        if self.augment and torch.rand(()).item() < 0.5:
            hr_image = TF.hflip(hr_image)

        lr_image = hr_image.resize(
            (self.lr_size, self.lr_size),
            resample=Image.Resampling.BICUBIC,
        )

        return TF.to_tensor(lr_image), TF.to_tensor(hr_image)


def make_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    *,
    persistent_workers: bool = False,
) -> DataLoader:
    """Grupuj pary w partie, losując kolejność tylko wtedy, gdy jest to wymagane."""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
        # trening odtwarza procesy co epokę, a ocena może je zachować między modelami.
        persistent_workers=persistent_workers and num_workers > 0,
    )


def read_split_info(path: Path) -> dict[str, object]:
    """Odczytaj pochodzenie i tożsamość utrwalonych części zbioru."""

    info = json.loads((path / "split_info.json").read_text(encoding="utf-8"))

    if info.get("format_version") != 1 or set(info["splits"]) != {
        "train",
        "val",
        "test",
    }:
        raise ValueError("Niepoprawny opis podziału danych.")

    return info


def load_partitions(path: Path) -> DatasetDict:
    """Wczytaj gotowy podział, sprawdzając zapisane liczebności i fingerprinty."""

    info = read_split_info(path)
    parts = load_from_disk(str(path))

    if not isinstance(parts, DatasetDict) or set(parts) != {"train", "val", "test"}:
        raise ValueError("Oczekiwano DatasetDict z częściami train, val i test.")

    for name, dataset in parts.items():
        actual = {"count": len(dataset), "fingerprint": dataset._fingerprint}

        if actual != info["splits"][name]:
            raise ValueError(f"Część {name} różni się od split_info.json.")
        if not {"image", "filename"}.issubset(dataset.column_names):
            raise ValueError(f"Brak obrazów lub nazw plików w części {name}.")

    return parts


def prepare_partitions(source_path: Path, output_path: Path, seed: int = 42) -> Path:
    """Zapisz podział matg raz, a później wykorzystuj jego zgodną gotową kopię."""

    source = load_wikiart(source_path)

    if output_path.exists():
        info = read_split_info(output_path)

        if (
            info["source_fingerprint"] != source._fingerprint
            or info["split_seed"] != seed
        ):
            raise ValueError(
                "Istniejący podział ma inne źródło lub seed; wybierz nowy katalog."
            )

        load_partitions(output_path)
        print(f"Wykorzystujemy zapisany podział: {display_path(output_path)}")
        return output_path

    temporary = output_path.with_name(output_path.name + ".tmp")

    if temporary.exists():
        raise FileExistsError(f"Pozostał niekompletny zapis: {display_path(temporary)}")

    train, val, test = split_wikiart(source, seed)
    parts = DatasetDict({"train": train, "val": val, "test": test})
    parts.save_to_disk(str(temporary))
    saved = load_from_disk(str(temporary))
    info = {
        "format_version": 1,
        "source_fingerprint": source._fingerprint,
        "source_count": len(source),
        "split_seed": seed,
        "split_method": "matg_80_10_10_v1",
        "splits": {
            name: {"count": len(part), "fingerprint": part._fingerprint}
            for name, part in saved.items()
        },
    }
    (temporary / "split_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    del saved
    temporary.rename(output_path)
    print(f"Zapisano podział: {display_path(output_path)}")

    return output_path


def select_limit(dataset: HFDataset, limit: int | None) -> HFDataset:
    """Wybierz początek utrwalonej części bez zmiany jej kolejności."""

    if limit is not None and limit <= 0:
        raise ValueError("Limit musi być dodatni albo równy None.")

    count = len(dataset) if limit is None else min(limit, len(dataset))

    return dataset.select(range(count))


def make_training_loaders(
    parts: DatasetDict, config: dict[str, object]
) -> tuple[DataLoader, DataLoader]:
    """Przygotuj świeże loadery treningu i walidacji roboczej dla jednego modelu."""

    scale = config["architecture"]["params"]["scale_factor"]
    train = SuperResolutionDataset(
        select_limit(parts["train"], config["train_limit"]),
        config["hr_size"],
        scale,
        augment=config["augment_train"],
    )
    val = SuperResolutionDataset(
        select_limit(parts["val"], config["val_limit"]), config["hr_size"], scale
    )

    return (
        make_loader(
            train, config["batch_size"], True, config["seed"], config["num_workers"]
        ),
        make_loader(
            val, config["eval_batch_size"], False, config["seed"], config["num_workers"]
        ),
    )
