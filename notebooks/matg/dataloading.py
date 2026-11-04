from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from preprocessing import preprocess_pipeline, remove_redundant_columns

# CONSTS

DATASET_NAME = "Artificio/WikiArt_Full"
CACHE_DIR = "data"
PROCESSED_PATH = Path("data/wikiart_train_processed")
DEFAULT_SEED = 42
BATCH_SIZE = 64

BASIC_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),
])


# DATASETS

def get_ds_full(processed_path: Path = PROCESSED_PATH) -> Dataset:
    """Load the preprocessed dataset from disk, building it on first use."""
    if (processed_path / "dataset_info.json").exists():
        return load_from_disk(str(processed_path))

    ds = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)
    ds = preprocess_pipeline(ds)
    ds = remove_redundant_columns(ds)
    ds.save_to_disk(str(processed_path))
    return ds


def get_splits(seed: int = DEFAULT_SEED, val_test_frac: float = 0.2):
    """Split into train / val / test (default 80 / 10 / 10)."""
    split = get_ds_full().train_test_split(test_size=val_test_frac, seed=seed)
    holdout = split["test"].train_test_split(test_size=0.5, seed=seed)
    return split["train"], holdout["train"], holdout["test"]


# DATALOADERS

def _make_loader(ds, transform, batch_size, shuffle, num_workers, generator=None):
    # `with_transform` receives a batch dict; v2 transforms apply to each image
    # in it, and default collate needs all images to be the same size.
    return DataLoader(
        ds.select_columns("image").with_transform(transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def get_splits_dataloaders(
    transform_train=BASIC_TRANSFORM,
    transform_val=BASIC_TRANSFORM,
    transform_test=BASIC_TRANSFORM,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 2,
    seed: int = DEFAULT_SEED,
):
    ds_train, ds_val, ds_test = get_splits(seed)
    g = torch.Generator().manual_seed(seed)

    return (
        _make_loader(ds_train, transform_train, batch_size, True, num_workers, g),
        _make_loader(ds_val, transform_val, batch_size, False, num_workers),
        _make_loader(ds_test, transform_test, batch_size, False, num_workers),
    )