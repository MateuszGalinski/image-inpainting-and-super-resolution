"""Dataset loading, splitting and DataLoader construction for WikiArt.

Everything configurable lives in `DataConfig`. Pass one config to every
function and the training and clustering notebooks are guaranteed to see
identical splits.

    cfg = DataConfig()
    train_dl, val_dl, test_dl = get_splits_dataloaders(cfg)

    # Same splits, but without shuffling (e.g. for encoding):
    train_dl, val_dl, test_dl = get_splits_dataloaders(replace(cfg, shuffle_train=False))
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from datasets import Dataset, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torch.nn.functional as F
from torchvision import tv_tensors
from torchvision.transforms.v2.functional import rgb_to_grayscale

from .preprocessing import preprocess_pipeline, remove_redundant_columns

#
# TRANSFORMS
#

IMG_SIZE = 128

RESIZE_128 = v2.Compose([
    v2.ToImage(),
    v2.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
])

BASIC_TRANSFORM = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True),  # uint8 [0, 255] -> float32 [0, 1]
])

class SobelEdges(v2.Transform):
    """
    RGB image (3, H, W) in [0, 1] -> Sobel edges (2, H, W), as in DeepCluster.

    Channels are horizontal and vertical gradients of the grayscale image.
    Raw responses lie in [-4, 4] and are rescaled to [0, 1] so a Sigmoid
    decoder can reconstruct them: 0.5 means no edge.
    """

    def transform(self, inpt, params):
        SOBEL_MAX_RESPONSE = 4.0
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=inpt.dtype)
        kernels = torch.stack([sobel_x, sobel_x.T]).unsqueeze(1)                  # (2, 1, 3, 3)

        gray = rgb_to_grayscale(inpt)                                             # (1, H, W)
        edges = F.conv2d(gray.unsqueeze(0), kernels, padding=1).squeeze(0)        # (2, H, W)
        return tv_tensors.wrap((edges + SOBEL_MAX_RESPONSE) / (2 * SOBEL_MAX_RESPONSE), like=inpt) # type: ignore


SOBEL_TRANSFORM = v2.Compose([*BASIC_TRANSFORM.transforms, SobelEdges()])


#
# CONFIG
#

@dataclass(frozen=True)
class DataConfig:
    """
    Immutable settings for dataset loading, splitting and batching.

    Frozen so a stray assignment in one notebook cell cannot silently change
    the splits for later cells. To get a variant, use
    `dataclasses.replace(cfg, field=value)`.

    Attributes:
        dataset_name: HuggingFace dataset identifier.
        cache_dir: Directory where the raw HuggingFace download is cached.
        processed_path: Directory where the preprocessed dataset is saved
            (via `save_to_disk`) and reloaded from on later runs.
        seed: Seed for splitting and for the train-loader shuffle.
        val_size: Fraction (0-1) of the full dataset used for validation.
        test_size: Fraction (0-1) of the full dataset used for testing.
            Train gets the remainder, i.e. 1 - val_size - test_size.
        batch_size: Samples per batch, for every loader.
        num_workers: DataLoader worker processes. 0 loads in the main process.
        shuffle_train: Whether the train loader shuffles. Validation and test
            loaders never shuffle.
        transform_train: Callable applied to train batches.
        transform_val: Callable applied to validation batches.
        transform_test: Callable applied to test batches.
        transform_full: Callable applied by `get_full_dataloader`.
    """

    dataset_name: str = "Artificio/WikiArt_Full"
    cache_dir: str = "data"
    processed_path: Path = Path("data/wikiart_train_processed")

    seed: int = 42
    val_size: float = 0.1
    test_size: float = 0.1

    batch_size: int = 64
    num_workers: int = 2
    shuffle_train: bool = True

    transform_train: Callable = BASIC_TRANSFORM
    transform_val: Callable = BASIC_TRANSFORM
    transform_test: Callable = BASIC_TRANSFORM
    transform_full: Callable = BASIC_TRANSFORM

    def __post_init__(self):
        if self.val_size <= 0 or self.test_size <= 0:
            raise ValueError("val_size and test_size must be positive")
        if self.val_size + self.test_size >= 1:
            raise ValueError("val_size + test_size must be below 1")


#
# DATASETS
#

def get_ds_full(cfg: DataConfig = DataConfig()) -> Dataset:
    """
    Load the preprocessed dataset, building it on first use.

    If `cfg.processed_path` already holds a saved dataset it is loaded from
    disk. Otherwise the raw dataset is downloaded, run through
    `preprocess_pipeline`, stripped of redundant columns and saved there, so
    the expensive steps only happen once.

    Args:
        cfg (DataConfig): Source, cache and output locations.

    Returns:
        Dataset: The preprocessed dataset (all rows, before any splitting).
    """
    if (cfg.processed_path / "dataset_info.json").exists():
        return load_from_disk(str(cfg.processed_path))  # type: ignore

    ds = load_dataset(cfg.dataset_name, split="train", cache_dir=cfg.cache_dir)
    ds = preprocess_pipeline(ds)
    ds = remove_redundant_columns(ds)
    ds.save_to_disk(str(cfg.processed_path))
    return ds


def get_splits(cfg: DataConfig = DataConfig()) -> tuple[Dataset, Dataset, Dataset]:
    """
    Split the dataset into train, validation and test sets.

    Done in two steps: first train vs. holdout (`val_size + test_size`), then
    the holdout is divided into validation and test. Both steps use
    `cfg.seed`, so the result is reproducible. With the defaults (0.1 / 0.1)
    the split is 80 / 10 / 10.

    Args:
        cfg (DataConfig): Provides `seed`, `val_size` and `test_size`.

    Returns:
        tuple[Dataset, Dataset, Dataset]: (train, validation, test).
    """
    holdout_size = cfg.val_size + cfg.test_size

    split = get_ds_full(cfg).train_test_split(test_size=holdout_size, seed=cfg.seed)
    holdout = split["test"].train_test_split(
        test_size=cfg.test_size / holdout_size, seed=cfg.seed
    )
    return split["train"], holdout["train"], holdout["test"]


#
# DATALOADERS
#

def _make_loader(
    ds: Dataset,
    transform: Callable,
    cfg: DataConfig,
    shuffle: bool,
) -> DataLoader:
    """
    Build an image-only DataLoader over `ds`.

    Only the 'image' column is kept, so metadata is never loaded. `transform`
    is applied lazily with `with_transform`: it receives a batch dict, and
    v2 transforms are applied to each image in it. Default collation needs
    all images to be the same size.

    The shuffle order is seeded from `cfg.seed` via a dedicated generator, so
    it does not depend on global RNG state. Persistent workers are only
    enabled when `num_workers > 0`, since PyTorch rejects them otherwise.

    Args:
        ds (Dataset): Dataset to iterate over.
        transform (Callable): Transform applied to each batch of images.
        cfg (DataConfig): Provides batch size, workers and seed.
        shuffle (bool): Whether to shuffle each epoch.

    Returns:
        DataLoader: Yields dicts with an 'image' tensor of shape (B, C, H, W).
    """
    generator = torch.Generator().manual_seed(cfg.seed) if shuffle else None

    return DataLoader(
        ds.select_columns("image").with_transform(transform), # type: ignore
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def get_splits_dataloaders(
    cfg: DataConfig = DataConfig(),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train, validation and test loaders.

    Only the train loader may shuffle (`cfg.shuffle_train`). Set it to False
    when the loader is used for encoding rather than training, so that batch
    order matches dataset order.

    Args:
        cfg (DataConfig): Split sizes, batching and per-split transforms.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: (train, validation, test).
    """
    ds_train, ds_val, ds_test = get_splits(cfg)

    return (
        _make_loader(ds_train, cfg.transform_train, cfg, cfg.shuffle_train),
        _make_loader(ds_val, cfg.transform_val, cfg, shuffle=False),
        _make_loader(ds_test, cfg.transform_test, cfg, shuffle=False),
    )


def get_full_dataloader(
    cfg: DataConfig = DataConfig(),
) -> tuple[Dataset, DataLoader]:
    """
    Return the full dataset together with an unshuffled loader over all of it.

    The loader never shuffles (`cfg.shuffle_train` is ignored), so batch i
    corresponds to rows [i * batch_size, ...) of the returned dataset and
    embeddings stay aligned with the metadata.

    Args:
        cfg (DataConfig): Batching and `transform_full`.

    Returns:
        tuple[Dataset, DataLoader]: The dataset and its loader.
    """
    ds = get_ds_full(cfg)
    return ds, _make_loader(ds, cfg.transform_full, cfg, shuffle=False)

#
# Creating dataloders for damaged images
#

from torch.utils.data import ConcatDataset, Dataset as TorchDataset

from restore_pipeline.damage.damage import DamageFn, PairTransform, random_ellipse_damage


def _loader_from_dataset(dataset: TorchDataset, cfg: DataConfig, shuffle: bool) -> DataLoader:
    """DataLoader boilerplate shared by every loader. `_make_loader` should call this too."""
    generator = torch.Generator().manual_seed(cfg.seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def get_damaged_splits_dataloaders(
    cfg: DataConfig = DataConfig(),
    damage_fn: DamageFn = random_ellipse_damage,
    mix: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Like `get_splits_dataloaders`, but batches are {'image': damaged, 'target': clean}.

    Each split uses its transform from `cfg` (transform_train / _val / _test)
    for both image and target; damage is applied after the transform.

    Args:
        cfg (DataConfig): Splits, batching and per-split transforms (same
            splits as the clean loaders).
        damage_fn (DamageFn): Damage applied to every damaged sample.
        mix (bool): False -> damaged samples only. True -> every clean image is
            also included as a clean pair (image == target), so the dataset has
            twice as many samples.

    Training damage is random on every read; validation and test damage is
    fixed per image, so their losses are comparable across epochs. The mix
    applies to every split, so validation matches the training distribution.
    """
    def build(ds: Dataset, transform: Callable, deterministic: bool, shuffle: bool) -> DataLoader:
        images = ds.select_columns("image")
        damaged = images.with_transform(
            PairTransform(transform, damage_fn, damage=True, deterministic=deterministic)
        )
        if mix:
            clean = images.with_transform(PairTransform(transform, damage=False))
            return _loader_from_dataset(ConcatDataset([clean, damaged]), cfg, shuffle)  # type: ignore
        return _loader_from_dataset(damaged, cfg, shuffle)  # type: ignore

    ds_train, ds_val, ds_test = get_splits(cfg)
    return (
        build(ds_train, cfg.transform_train, deterministic=False, shuffle=cfg.shuffle_train),
        build(ds_val, cfg.transform_val, deterministic=True, shuffle=False),
        build(ds_test, cfg.transform_test, deterministic=True, shuffle=False),
    )