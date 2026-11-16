import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from typing import cast

import pandas as pd
from datasets import Dataset

#
# CONSTANTS
#

COMPLETENESS_COLUMNS = ("title", "artist", "date", "genre", "style")
LABEL_COLUMNS = ("artist", "date", "genre", "style", "description")
DEFAULT_NORMALIZATION_COLS = ("title", "artist", "date", "genre", "style")

# Columns that are not needed for image-only training. Used by the standalone
# remove_redundant_columns(), NOT by preprocess_pipeline().
REDUNDANT_COLUMNS = (
    "description",
    "embeddings_pca512",
    "resnet50_non_robust_feats",
    "resnet50_robust_feats",
)

# Longer alternatives first, and a digit must follow, so "ca. 1900" is not
# half-stripped to "a. 1900".
CIRCA_RE = re.compile(r"^(?:circa|approx\.?|ca\.?|c\.?)\s*(?=\d)")
FLOAT_YEAR_RE = re.compile(r"^(\d{1,4})\.0$")


#
# NORMALIZATION PIPELINE
#

def normalize_value(value: str | None, column: str) -> str | None:
    """
    Normalize a single metadata value.

    Applies Unicode NFKC normalization, collapses whitespace (including
    non-breaking spaces) and casefolds the text. Blank strings become None.
    For the 'date' column it also strips 'circa' prefixes (e.g. 'c.', 'ca.',
    'approx.') and turns float-like years ('1900.0') into plain years ('1900').

    Args:
        value (str | None): The raw value. Can be None.
        column (str): The column the value belongs to, used for column-specific rules.

    Returns:
        str | None: The normalized string, or None if the value is missing or blank.
    """
    if value is None:
        return None

    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.split()).casefold()

    if not text:
        return None

    if column == "date":
        text = CIRCA_RE.sub("", text)
        text = FLOAT_YEAR_RE.sub(r"\1", text)

    return text


def normalize_dataset(
    dataset: Dataset,
    columns: tuple[str, ...] = DEFAULT_NORMALIZATION_COLS,
) -> Dataset:
    """
    Apply text normalization to the given columns.

    Uses `input_columns` so that only the listed columns are read. Images and
    the large feature columns are not decoded or re-encoded.

    Args:
        dataset (Dataset): The dataset containing the columns to normalize.
        columns (tuple[str, ...]): Names of the columns to process.

    Returns:
        Dataset: A new dataset with the specified columns normalized.
    """
    cols = list(columns)

    def normalize_row(*values):
        return {col: normalize_value(val, col) for col, val in zip(cols, values)}

    return dataset.map(
        normalize_row,
        input_columns=cols,
        features=dataset.features,
    )


#
# DEDUPLICATION PIPELINE
#

def find_duplicate_groups(dataset: Dataset) -> list[list[int]]:
    """
    Identify groups of exact duplicate images.

    Uses SHA-256 on the raw image bytes. Images are decoded one at a time
    instead of loading the whole column into memory.

    Args:
        dataset (Dataset): A dataset containing an 'image' column.

    Returns:
        list[list[int]]: Index groups. Each inner list holds the indices of
            images that are exact duplicates of one another.
    """
    hash_to_indices: defaultdict[str, list[int]] = defaultdict(list)

    for idx, row in enumerate(dataset.select_columns(["image"])):
        digest = hashlib.sha256(row["image"].tobytes()).hexdigest() # type: ignore
        hash_to_indices[digest].append(idx)

    return [indices for indices in hash_to_indices.values() if len(indices) > 1]


def identify_drop_candidates(
    dataset: Dataset,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """
    Decide which duplicates to drop and which kept records have unreliable labels.

    For each group of identical images, the keeper is the record with the fewest
    missing fields (see `COMPLETENESS_COLUMNS`). Ties are broken randomly with a
    seeded RNG. The other records in the group are dropped. If the artists in a
    group disagree, the keeper is flagged as having unreliable labels.

    Artists are compared exactly as stored, so the dataset must already be
    normalized (see `normalize_dataset`); otherwise case, whitespace or NBSP
    differences would be reported as conflicts.

    Args:
        dataset (Dataset): The normalized dataset containing images and metadata.
        seed (int, optional): Seed for reproducible tie-breaking. Defaults to 42.

    Returns:
        tuple[set[int], set[int]]:
            - drop: Indices of duplicate records to remove.
            - label_unreliable: Indices of kept records whose group had
              conflicting artists.
    """
    rng = random.Random(seed)
    exact_groups = find_duplicate_groups(dataset)

    # Only the small metadata columns; avoids loading the embedding columns.
    meta = cast(
        pd.DataFrame,
        dataset.select_columns(list(COMPLETENESS_COLUMNS)).to_pandas(),
    )
    meta["n_missing"] = meta[list(COMPLETENESS_COLUMNS)].isna().sum(axis=1)

    drop: set[int] = set()
    label_unreliable: set[int] = set()

    for group in sorted(exact_groups):
        fewest_missing = meta.loc[group, "n_missing"].min()
        tied = [idx for idx in group if meta.at[idx, "n_missing"] == fewest_missing]
        keeper = rng.choice(tied)
        drop.update(idx for idx in group if idx != keeper)

        # More than one distinct non-null artist -> the record is likely wrong.
        if meta.loc[group, "artist"].dropna().nunique() > 1:
            label_unreliable.add(keeper)

    return drop, label_unreliable


def null_unreliable_records(
    dataset: Dataset,
    label_unreliable: set[int],
) -> Dataset:
    """
    Set the label columns to None for records flagged as unreliable.

    Args:
        dataset (Dataset): The dataset to process.
        label_unreliable (set[int]): Indices of records whose labels should be nulled.

    Returns:
        Dataset: A new dataset with `LABEL_COLUMNS` set to None for those records.
    """
    def null_labels(artist, date, genre, style, description, idx):
        if idx in label_unreliable:
            return dict.fromkeys(LABEL_COLUMNS, None)

        return {
            "artist": artist,
            "date": date,
            "genre": genre,
            "style": style,
            "description": description,
        }

    return dataset.map(
        null_labels,
        input_columns=list(LABEL_COLUMNS),
        with_indices=True,
        features=dataset.features,
    )


def deduplicate_dataset(dataset: Dataset, seed: int = 42) -> Dataset:
    """
    Deduplicate the dataset and null labels of records with conflicting metadata.

    Expects an already normalized dataset (see `normalize_dataset`), because
    the artist conflict check compares strings exactly.

    Steps:
    1. Find duplicate images via hashing.
    2. Choose a keeper per group (fewest missing fields, seeded random tie-break)
       and flag keepers whose group has conflicting artists.
    3. Null the labels of the flagged keepers.
    4. Select only the kept records.

    Args:
        dataset (Dataset): The normalized dataset to deduplicate.
        seed (int, optional): Seed for tie-breaking. Defaults to 42.

    Returns:
        Dataset: The deduplicated dataset.
    """
    drop, label_unreliable = identify_drop_candidates(dataset, seed=seed)
    keep = [idx for idx in range(len(dataset)) if idx not in drop]

    cleaned = null_unreliable_records(dataset, label_unreliable)
    return cleaned.select(keep)


#
# FINAL PREPROCESS FUNCTION
#

def preprocess_pipeline(
    dataset: Dataset,
    columns: tuple[str, ...] = DEFAULT_NORMALIZATION_COLS,
    seed: int = 42,
) -> Dataset:
    """
    Run the full preprocessing pipeline.

    Normalization must run before deduplication: it turns blank strings into
    None (so completeness is counted correctly) and removes case, whitespace
    and NBSP differences (so they are not mistaken for artist conflicts). The
    input dataset is not modified.

    Pipeline steps:
    1. Normalize: NFKC, whitespace, casefold, blank -> None, date cleanup.
    2. Deduplicate: drop exact duplicate images, keep the most complete record,
       null labels where the artists conflict.

    Args:
        dataset (Dataset): The raw input dataset.
        columns (tuple[str, ...], optional): Metadata columns to normalize.
        seed (int, optional): Seed for deduplication tie-breaking. Defaults to 42.

    Returns:
        Dataset: The normalized and deduplicated dataset.
    """
    dataset = normalize_dataset(dataset, columns=columns)
    dataset = deduplicate_dataset(dataset, seed=seed)
    return dataset


#
# STANDALONE HELPERS (not part of preprocess_pipeline)
#

def remove_redundant_columns(
    dataset: Dataset,
    columns: tuple[str, ...] = REDUNDANT_COLUMNS,
) -> Dataset:
    """
    Remove columns that are not needed for image-only training.

    This is deliberately NOT part of `preprocess_pipeline`. Call it separately
    once you know you no longer need these columns. Columns that are not
    present in the dataset are ignored, so it is safe to call more than once.

    Args:
        dataset (Dataset): The dataset to remove columns from.
        columns (tuple[str, ...], optional): Columns to remove. Defaults to
            `REDUNDANT_COLUMNS` (description, embeddings_pca512, and both
            ResNet50 feature columns).

    Returns:
        Dataset: A new dataset without the specified columns.
    """
    to_remove = [col for col in columns if col in dataset.column_names]
    return dataset.remove_columns(to_remove)