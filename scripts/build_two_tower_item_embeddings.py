"""
Build dense item-content embeddings for Two-Tower from item metadata.

Pipeline (advisor-confirmed):
    item text -> TF-IDF -> TruncatedSVD -> dense embedding (64 dims by default)

This is a GLOBAL, per-dataset embedding -- item text does not change across
sparsification variants (random/head/tail keep_frac), so it is built once per
dataset, not once per experiment folder. Two-Tower training itself is NOT
implemented here; this script only produces the item content embedding table.

Content fields mirror scripts/build_vsm_item_attributes.py exactly (via the
shared scripts/content_utils.py helpers), so VSM and Two-Tower are built from
the same underlying text -- a fair comparison between the two item
representations:
    H&M:    prod_name + detail_desc
    Amazon: title + description

Usage:
    python scripts/build_two_tower_item_embeddings.py --dataset hm
    python scripts/build_two_tower_item_embeddings.py --dataset amazon
    python scripts/build_two_tower_item_embeddings.py --dataset amazon --n-components 64
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from content_utils import (
    DATASET_PROFILES,
    MAX_FEATURES,
    MIN_DF,
    TOKEN_PATTERN,
    build_doc_column,
    dedup_by_item_key,
    drop_null_item_ids,
    get_dataset_profile,
    load_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "item_embeddings"

N_COMPONENTS = 64
# Matches the seed convention already used across configs/recbole/*.yml.
RANDOM_STATE = 2020


# ── Step A: metadata -> one document per item ─────────────────────────────────

def _prepare_documents(
    metadata_path: Path, item_key: str, text_cols: list[str], meta_dtype: dict
) -> pd.DataFrame:
    """Load metadata and build exactly one text `doc` per item.

    Reuses the same shared steps as VSM (scripts/content_utils.py), plus an
    explicit null-item-id check this script requires as a sanity gate.
    """
    meta = load_metadata(metadata_path, meta_dtype)
    meta = drop_null_item_ids(meta, item_key)
    meta = dedup_by_item_key(meta, item_key)
    meta = build_doc_column(meta, text_cols)
    return meta


# ── Step B: TF-IDF ────────────────────────────────────────────────────────────

def _vectorize_tfidf(docs: pd.Series, min_df: int, max_features: int):
    """Returns (X sparse matrix, vocab list). Raises ValueError with a clear
    message on an empty vocabulary, mirroring build_vsm_item_attributes.py's
    _vectorize().

    Settings (token_pattern, ngram_range, min_df, max_features) are reused
    as-is from the VSM CountVectorizer so the two item representations are
    built from directly comparable tokenization. norm/use_idf/smooth_idf/
    sublinear_tf are left at TfidfVectorizer defaults -- the task only asked
    to mirror the listed tokenization settings, and these are standard,
    well-understood TF-IDF defaults with no VSM equivalent to match against.
    """
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_features=max_features,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
    )
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError as exc:
        if "empty vocabulary" in str(exc) or "max_df corresponds to fewer documents" in str(exc):
            raise ValueError(
                f"TfidfVectorizer produced an empty vocabulary "
                f"(min_df={min_df} may be too high for {len(docs)} items). "
                f"Try: --min-df 1"
            ) from None
        raise
    return X, list(vectorizer.get_feature_names_out())


# ── Step C: TruncatedSVD ──────────────────────────────────────────────────────

def _reduce_svd(X, n_components: int, random_state: int):
    """Returns (dense embedding ndarray, explained_variance_ratio sum)."""
    max_components = min(X.shape) - 1
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} is too large for TF-IDF matrix of "
            f"shape {X.shape} (max usable: {max_components}). "
            f"Try a smaller --n-components."
        )
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embedding = svd.fit_transform(X)
    return embedding, float(svd.explained_variance_ratio_.sum())


# ── Step D: build + validate output table ─────────────────────────────────────

def build_item_embeddings(
    metadata_path: Path,
    item_key: str,
    item_dtype,
    text_cols: list[str],
    meta_dtype: dict,
    n_components: int = N_COMPONENTS,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    meta = _prepare_documents(metadata_path, item_key, text_cols, meta_dtype)
    if meta.empty:
        raise ValueError(f"No items found in {metadata_path}")

    # Sanity: item_key unique after preparation (drop_null_item_ids +
    # dedup_by_item_key already enforce this; assert as an explicit gate).
    n_items = len(meta)
    assert meta[item_key].isna().sum() == 0, "BUG: null item_key survived preparation"
    assert meta[item_key].duplicated().sum() == 0, "BUG: duplicate item_key survived preparation"

    n_empty_doc = int((meta["doc"].str.strip() == "").sum())
    print(f"  Items after cleaning:    {n_items}")
    print(f"  Items with empty text:   {n_empty_doc}")
    if n_empty_doc > 0:
        print(
            "  [STRATEGY] Empty-text items are KEPT in the output (every item "
            "must have an embedding for full Two-Tower item coverage). Their "
            "TF-IDF row is all-zero, so TruncatedSVD (a linear projection) "
            "maps them to the all-zero embedding vector -- these rows carry "
            "no content signal and should not be treated as meaningful for "
            "content similarity."
        )

    X, vocab = _vectorize_tfidf(meta["doc"], min_df=min_df, max_features=max_features)
    print(f"  TF-IDF matrix shape:     {X.shape}  (items x vocabulary)")
    print(f"  Vocabulary size:         {len(vocab)}")

    embedding, explained_var = _reduce_svd(X, n_components=n_components, random_state=random_state)
    print(f"  SVD n_components:        {n_components}")
    print(f"  SVD explained variance:  {explained_var:.4f}  (sum of explained_variance_ratio_)")

    # Sanity: embedding shape / finiteness.
    assert embedding.shape == (n_items, n_components), (
        f"BUG: embedding shape {embedding.shape} != expected ({n_items}, {n_components})"
    )
    n_nonfinite = int((~np.isfinite(embedding)).sum())
    if n_nonfinite > 0:
        raise RuntimeError(
            f"Embedding contains {n_nonfinite} NaN/Inf value(s) -- refusing to save."
        )

    item_ids = meta[item_key].apply(item_dtype)
    emb_cols = [f"emb_{i}" for i in range(1, n_components + 1)]
    out = pd.DataFrame(embedding, columns=emb_cols)
    out.insert(0, "item_id", item_ids.values)

    # Sanity: output row count == unique items used to build documents, no
    # duplicate item rows in the output.
    assert len(out) == n_items, (
        f"BUG: output rows {len(out)} != items used to build documents {n_items}"
    )
    n_out_dup = int(out["item_id"].duplicated().sum())
    if n_out_dup > 0:
        raise RuntimeError(f"BUG: {n_out_dup} duplicate item rows in output -- refusing to save.")

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    valid_datasets = ", ".join(DATASET_PROFILES)
    p = argparse.ArgumentParser(
        description="Build TF-IDF -> TruncatedSVD dense item-content embeddings for Two-Tower.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", required=True, choices=list(DATASET_PROFILES),
        help=f"Dataset profile: {valid_datasets}",
    )
    p.add_argument(
        "--metadata-path", default=None, type=Path,
        help="Path to metadata CSV. Defaults to the dataset profile's conventional raw path.",
    )
    p.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    p.add_argument(
        "--output-path", default=None, type=Path,
        help="Full output .parquet path (overrides --output-dir + default filename).",
    )
    p.add_argument(
        "--n-components", type=int, default=N_COMPONENTS,
        help=f"TruncatedSVD output dimensionality (default: {N_COMPONENTS})",
    )
    p.add_argument(
        "--min-df", type=int, default=MIN_DF,
        help=f"TfidfVectorizer min_df (default: {MIN_DF}, same as VSM)",
    )
    p.add_argument(
        "--max-features", type=int, default=MAX_FEATURES,
        help=f"TfidfVectorizer max_features (default: {MAX_FEATURES}, same as VSM)",
    )
    p.add_argument(
        "--random-state", type=int, default=RANDOM_STATE,
        help=f"TruncatedSVD random_state (default: {RANDOM_STATE})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    profile = get_dataset_profile(args.dataset)

    metadata_path = args.metadata_path or profile["default_metadata_path"]
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise SystemExit(f"[ERROR] Metadata file not found: {metadata_path}")

    if args.output_path is not None:
        output_path = Path(args.output_path)
    else:
        output_path = Path(args.output_dir) / f"{args.dataset}_tfidf_svd{args.n_components}.parquet"

    print(f"\n[BUILD TWO-TOWER ITEM EMBEDDINGS] dataset={args.dataset}")
    print(f"  Metadata source:         {metadata_path}")

    out = build_item_embeddings(
        metadata_path=metadata_path,
        item_key=profile["item_key"],
        item_dtype=profile["item_dtype"],
        text_cols=profile["text_cols"],
        meta_dtype=profile["meta_dtype"],
        n_components=args.n_components,
        min_df=args.min_df,
        max_features=args.max_features,
        random_state=args.random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"  Output rows:             {len(out)}")
    print(f"  Output dims:             {args.n_components}")
    print(f"  Output path:             {output_path}")


if __name__ == "__main__":
    main()
