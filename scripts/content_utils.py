"""
Shared item-content/text-preparation utilities.

Used by both:
  - scripts/build_vsm_item_attributes.py  (Elliot VSM, CountVectorizer -> TSV)
  - scripts/build_two_tower_item_embeddings.py  (TF-IDF -> TruncatedSVD -> parquet)

Only the parts genuinely shared by both consumers live here: dataset
profiles, metadata loading, safe text coercion, document assembly, and
defensive dedup to one row per item. Vectorization, dimensionality
reduction, and output formats are consumer-specific and stay in their own
scripts.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# ── Shared vectorizer-preprocessing constants ─────────────────────────────────
# Reused as-is by both CountVectorizer (VSM) and TfidfVectorizer (Two-Tower)
# so the two representations are built from directly comparable tokenization.
MIN_DF = 2
MAX_FEATURES = 50_000
TOKEN_PATTERN = r"(?u)[a-zA-Z0-9]+"

# ── Dataset profiles ─────────────────────────────────────────────────────────
# item_key            : column name for item identifier in metadata CSV
# item_dtype          : Python type for item IDs (int for H&M, str for Amazon)
# text_cols           : columns concatenated into the item document (`doc`)
# meta_dtype          : dtype hints for pd.read_csv on the metadata file
# default_metadata_path : conventional raw metadata location for this dataset
DATASET_PROFILES: dict[str, dict] = {
    "hm": {
        "item_key": "article_id",
        "item_dtype": int,
        "text_cols": ["prod_name", "detail_desc"],
        "meta_dtype": {"article_id": int},
        "default_metadata_path": RAW_DIR / "metadata_hm.csv",
    },
    "amazon": {
        "item_key": "parent_asin",
        "item_dtype": str,
        "text_cols": ["title", "description"],
        "meta_dtype": {},
        "default_metadata_path": RAW_DIR / "metadata_amazon.csv",
    },
}


def get_dataset_profile(dataset: str) -> dict:
    if dataset not in DATASET_PROFILES:
        raise ValueError(
            f"Unknown dataset profile: {dataset!r}. Valid: {sorted(DATASET_PROFILES)}"
        )
    return DATASET_PROFILES[dataset]


# ── Metadata loading ────────────────────────────────────────────────────────

def load_metadata(metadata_path: Path, meta_dtype: dict | None = None) -> pd.DataFrame:
    """Load metadata CSV and strip a leading UTF-8 BOM from column names."""
    meta_dtype = meta_dtype or {}
    kw = {"dtype": meta_dtype} if meta_dtype else {}
    meta = pd.read_csv(metadata_path, **kw)
    meta.columns = [c.lstrip("﻿") for c in meta.columns]
    return meta


# ── Text handling ────────────────────────────────────────────────────────────

def safe_str(val) -> str:
    """Convert a value to string -- handles NaN, list, dict safely."""
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    if isinstance(val, dict):
        return " ".join(str(v) for v in val.values())
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def build_doc_column(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    """Concatenate text_cols into a single `doc` column (missing cols -> "")."""
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(safe_str)
        else:
            df[col] = ""
    df["doc"] = df[text_cols].agg(" ".join, axis=1).str.strip()
    return df


# ── Item-identity hygiene ────────────────────────────────────────────────────

def drop_null_item_ids(df: pd.DataFrame, item_key: str) -> pd.DataFrame:
    """Drop rows with a null item_key, printing a clear warning if any exist."""
    null_mask = df[item_key].isna()
    n_null = int(null_mask.sum())
    if n_null > 0:
        print(f"  [WARN] {n_null} metadata row(s) have a null '{item_key}' -- dropped.")
        df = df.loc[~null_mask].reset_index(drop=True)
    return df


def dedup_by_item_key(df: pd.DataFrame, item_key: str) -> pd.DataFrame:
    """Guarantee one row per item_key: keep the first occurrence, warn on drops.

    Defensive only -- both current datasets already have a unique item_key,
    so this is a no-op unless the source metadata changes.
    """
    dup_mask = df[item_key].duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        print(
            f"  [WARN] {n_dup} duplicate metadata row(s) for column "
            f"'{item_key}' -- keeping first occurrence per item, dropping the rest."
        )
        df = df.loc[~dup_mask].reset_index(drop=True)
    return df


def load_item_documents(
    metadata_path: Path,
    item_key: str,
    text_cols: list[str],
    meta_dtype: dict | None = None,
) -> pd.DataFrame:
    """Load metadata, build the `doc` text column, and guarantee one row per
    item_key (defensive dedup: keep the first occurrence, warn if any
    duplicates were found and dropped).

    Convenience wrapper over load_metadata + dedup_by_item_key + build_doc_column
    for the common "no pre-filtering" case (global item set). Callers that need
    to filter rows first (e.g. to a specific train/test item set) should call
    load_metadata / dedup_by_item_key / build_doc_column directly, in that order,
    around their own filtering step.

    Returns the full metadata frame (all original columns) plus `doc`.
    """
    meta = load_metadata(metadata_path, meta_dtype)
    meta = dedup_by_item_key(meta, item_key)
    meta = build_doc_column(meta, text_cols)
    return meta
