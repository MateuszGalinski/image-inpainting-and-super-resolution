from collections import defaultdict
import hashlib
import random

import pandas as pd
from datasets import Dataset

import re
import unicodedata

from typing import cast

#
# DEDUPLICATION PIPELINE
#

COMPLETENESS_COLUMNS = ("title", "artist", "date", "genre", "style")
LABEL_COLUMNS = ("artist", "date", "genre", "style", "description")


def find_duplicate_groups(dataset: Dataset) -> list[list[int]]:
    """
    Identify groups of exact duplicate images in the dataset.

    Uses SHA-256 hashing on the image bytes to find exact pixel-level duplicates.

    Args:
        dataset (Dataset): A Hugging Face Dataset containing an 'image' column.

    Returns:
        list[list[int]]: A list of index groups. Each inner list contains the indices 
            of images that are exact duplicates of one another.
    """
    hash_to_indices: defaultdict[str, list[int]] = defaultdict(list)

    for idx, image in enumerate(dataset["image"]):
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        hash_to_indices[digest].append(idx)

    return [
        indices
        for indices in hash_to_indices.values()
        if len(indices) > 1
    ]


def identify_drop_candidates(
    dataset: Dataset,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """
    Determine which duplicates to drop and identify kept records with conflicting metadata.

    For each group of identical images, this function selects a single "keeper" record 
    that has the fewest missing metadata fields (based on `COMPLETENESS_COLUMNS`). Ties 
    are broken randomly. Remaining duplicates are marked to be dropped. If the duplicate 
    group has conflicting 'artist' metadata, the keeper is marked as having unreliable labels.

    Args:
        dataset (Dataset): The dataset containing images and metadata.
        seed (int, optional): Random seed for reproducible tie-breaking. Defaults to 42.

    Returns:
        tuple[set[int], set[int]]: A tuple containing two sets:
            - drop: Indices of duplicate records to be removed.
            - label_unreliable: Indices of kept records with conflicting artist labels 
              among their duplicates.
    """
    rng = random.Random(seed)
    exact_groups = find_duplicate_groups(dataset)

    meta = cast(pd.DataFrame, dataset.remove_columns("image").to_pandas())
    meta["n_missing"] = meta[list(COMPLETENESS_COLUMNS)].isna().sum(axis=1)

    drop: set[int] = set()
    label_unreliable: set[int] = set()

    for group in sorted(exact_groups):
        fewest_missing = meta.loc[group, "n_missing"].min()
        tied = [
            idx for idx in group
            if meta.at[idx, "n_missing"] == fewest_missing
        ]
        keeper = rng.choice(tied)
        drop.update(idx for idx in group if idx != keeper)

        if meta.loc[group, "artist"].dropna().nunique() > 1:
            label_unreliable.add(keeper)

    return drop, label_unreliable


def null_unreliable_records(
    dataset: Dataset,
    label_unreliable: set[int],
) -> Dataset:
    """
    Clear label columns for records that have been flagged as unreliable.

    Replaces the values in `LABEL_COLUMNS` with `None` for all records whose 
    index is present in the `label_unreliable` set.

    Args:
        dataset (Dataset): The dataset to process.
        label_unreliable (set[int]): A set of indices corresponding to records 
            whose labels should be nulled.

    Returns:
        Dataset: A new dataset with the target labels set to `None`.
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
    Deduplicate the dataset and clean up unreliable metadata.

    This function orchestrates the deduplication pipeline:
    1. Identifies duplicate images via hashing.
    2. Determines which duplicates to drop (favoring completeness) and which 
       remaining records have conflicting artist data.
    3. Nullifies label metadata for those conflicting records.
    4. Filters out the dropped records and returns the cleaned dataset.

    Args:
        dataset (Dataset): The dataset to deduplicate.
        seed (int, optional): Random seed for tie-breaking. Defaults to 42.

    Returns:
        Dataset: The cleaned and deduplicated dataset.
    """
    drop, label_unreliable = identify_drop_candidates(dataset, seed=seed)
    keep = [idx for idx in range(len(dataset)) if idx not in drop]

    cleaned = null_unreliable_records(dataset, label_unreliable)
    return cleaned.select(keep)

#
# NORMALIZATION PIPELINE
#

DEFAULT_NORMALIZATION_COLS = ("title", "artist", "date", "genre", "style")

def normalize_value(value: str | None, column: str) -> str | None:
    """
    Normalize a single metadata value.

    Applies Unicode NFKC normalization, collapses multiple spaces into a single space, 
    and converts text to lowercase. For the 'date' column specifically, it strips out 
    common 'circa' prefixes (e.g., 'c.', 'ca.', 'approx.') and formats float-like years 
    (e.g., '1900.0') back into standard integer strings.

    Args:
        value (str | None): The raw string value to normalize. Can be None.
        column (str): The name of the column this value belongs to, used to 
            trigger column-specific normalization rules.

    Returns:
        str | None: The cleaned and normalized string, or None if the input was None.
    """
    if value is None:
        return None
        
    text = unicodedata.normalize("NFKC", str(value))
    text = " ".join(text.split()).casefold()

    if column == "date":
        text = re.sub(r"^(?:c(?:irca)?\.?|ca\.?|approx\.?)\s*", "", text)
        text = re.sub(r"^(\d{1,4})\.0$", r"\1", text)

    return text

def normalize_dataset(dataset: Dataset, columns: tuple[str, ...] = DEFAULT_NORMALIZATION_COLS) -> Dataset:
    """
    Apply text normalization to specific columns across the dataset.

    Overwrites the original values in the specified columns with their cleaned 
    and normalized equivalents.

    Args:
        dataset (Dataset): The dataset containing the columns to normalize.
        columns (tuple[str, ...]): A tuple of column names that should be processed.

    Returns:
        Dataset: A new dataset with the specified columns updated to their 
            normalized values.
    """
    def normalize_row(row):
        return {
            col: normalize_value(row[col], col)
            for col in columns
        }
        
    return dataset.map(normalize_row)

#
# FINAL PREPROCESS FUNCTION
#

def preprocess_pipeline(dataset : Dataset, columns : tuple[str, ...] = DEFAULT_NORMALIZATION_COLS) -> Dataset:
    """
    Execute the full preprocessing pipeline on the dataset.

    This function sequentially applies deduplication and text normalization. 
    It does not modify the original dataset in place; instead, it returns 
    a new processed Dataset.

    Pipeline steps:
    1. Deduplicate: Removes exact duplicate images, keeping the record with 
       the most complete metadata, and nullifies labels for conflicting duplicates.
    2. Normalize: Cleans text data (NFKC normalization, lowercasing, whitespace 
       stripping) and standardizes date formats in the specified columns.

    Args:
        dataset (Dataset): The raw input dataset.
        columns (tuple[str, ...], optional): The metadata columns to normalize. 
            Defaults to DEFAULT_NORMALIZATION_COLS.

    Returns:
        Dataset: A new dataset that has been deduplicated and normalized.
    """
    dataset = deduplicate_dataset(dataset)
    dataset = normalize_dataset(dataset, columns=columns)
    return dataset