"""
Build VSM ItemAttributes file for Elliot from item metadata CSV.

Format (ItemAttributes loader in Elliot):
  item_id<TAB>feature_id_1<TAB>feature_id_2<...>
- Feature IDs start from 1.
- Item IDs preserve their original type: integers for H&M, strings for Amazon.

Có hai chế độ:

  --global-output PATH  (khuyến nghị)
      Build một file dùng chung từ toàn bộ metadata, không lọc theo folder experiment.

      python scripts/build_vsm_item_attributes.py \\
          --metadata-path data/raw/metadata_hm.csv \\
          --global-output data/processed/hm_item_attributes.tsv \\
          --dataset hm

      python scripts/build_vsm_item_attributes.py \\
          --metadata-path data/raw/metadata_amazon.csv \\
          --global-output data/processed/amazon_item_attributes.tsv \\
          --dataset amazon

  --dataset-dir / --all  (per-folder)
      Build riêng cho từng folder experiment, chỉ chứa item có trong train+test.

      python scripts/build_vsm_item_attributes.py \\
          --metadata-path data/raw/metadata_hm.csv \\
          --dataset-dir data/processed/hm_random_keep0.05 \\
          --dataset hm
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── Vectorizer defaults ──────────────────────────────────────────────────────
MIN_DF = 2
MAX_FEATURES = 50_000
TOKEN_PATTERN = r"(?u)[a-zA-Z0-9]+"

# ── Dataset profiles ─────────────────────────────────────────────────────────
# item_key   : column name for item identifier in metadata CSV
# item_dtype : Python type for item IDs (int for H&M, str for Amazon)
# text_cols  : columns to concatenate into VSM document
# meta_dtype : dtype hints for pd.read_csv on metadata file

_DATASET_PROFILES: dict[str, dict] = {
    "hm": {
        "item_key": "article_id",
        "item_dtype": int,
        "text_cols": ["prod_name", "detail_desc"],
        "meta_dtype": {"article_id": int},
    },
    "amazon": {
        "item_key": "parent_asin",
        "item_dtype": str,
        "text_cols": ["title", "description"],
        "meta_dtype": {},
    },
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_meta(metadata_path: Path, meta_dtype: dict) -> pd.DataFrame:
    kw = {"dtype": meta_dtype} if meta_dtype else {}
    meta = pd.read_csv(metadata_path, **kw)
    meta.columns = [c.lstrip("﻿") for c in meta.columns]  # strip UTF-8 BOM
    return meta


def _safe_str(val) -> str:
    """Convert a value to string — handles NaN, list, dict safely."""
    if isinstance(val, (list,)):
        return " ".join(str(v) for v in val)
    if isinstance(val, dict):
        return " ".join(str(v) for v in val.values())
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val)


def _build_doc_column(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(_safe_str)
        else:
            df[col] = ""
    df["doc"] = df[text_cols].agg(" ".join, axis=1).str.strip()
    return df


def _vectorize(docs: pd.Series, min_df: int, max_features: int) -> tuple:
    """Returns (X sparse matrix, vocab list). Raises ValueError with clear message on empty vocab."""
    vectorizer = CountVectorizer(
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
                f"CountVectorizer produced an empty vocabulary "
                f"(min_df={min_df} may be too high for {len(docs)} items). "
                f"Try: --min-df 1"
            ) from None
        raise
    return X, list(vectorizer.get_feature_names_out())


def _build_feature_map(X, df: pd.DataFrame, item_key: str, item_dtype) -> dict:
    """Build {item_id -> [feat_id, ...]} mapping with 1-based feature IDs."""
    feature_map: dict = {}
    rows, cols_idx = X.nonzero()
    for row, col in zip(rows, cols_idx):
        aid = item_dtype(df.at[row, item_key])
        feature_map.setdefault(aid, []).append(col + 1)  # 1-based feature IDs
    for row in range(len(df)):
        aid = item_dtype(df.at[row, item_key])
        feature_map.setdefault(aid, [])
    return feature_map


def _write_tsv(item_ids: list, feature_map: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item_id in item_ids:
            features = feature_map.get(item_id, [])
            if features:
                f.write("\t".join([str(item_id)] + [str(fid) for fid in sorted(features)]) + "\n")
            else:
                f.write(f"{item_id}\n")


def _write_vocab(vocab: list[str], output_path: Path) -> Path:
    vocab_path = output_path.with_name(output_path.stem + "_vocab.csv")
    pd.DataFrame({"feature_id": range(1, len(vocab) + 1), "token": vocab}).to_csv(
        vocab_path, index=False
    )
    return vocab_path


# ── Global mode ───────────────────────────────────────────────────────────────

def build_global_item_attributes(
    metadata_path: Path,
    output_path: Path,
    item_key: str,
    item_dtype,
    text_cols: list[str],
    meta_dtype: dict,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
) -> None:
    """Build item_attributes.tsv from ALL items in metadata (no per-experiment filtering)."""
    output_path = Path(output_path)

    meta = _load_meta(metadata_path, meta_dtype)
    meta = _build_doc_column(meta, text_cols)

    if meta.empty:
        raise ValueError(f"No items found in {metadata_path}")

    X, vocab = _vectorize(meta["doc"], min_df=min_df, max_features=max_features)
    feature_map = _build_feature_map(X, meta, item_key, item_dtype)

    item_ids = sorted(meta[item_key].apply(item_dtype).tolist())
    _write_tsv(item_ids, feature_map, output_path)
    vocab_path = _write_vocab(vocab, output_path)

    n_total = len(item_ids)
    n_empty = int((meta["doc"].str.strip() == "").sum())
    n_with_features = sum(1 for v in feature_map.values() if v)

    print(f"  Mode:                    global (all metadata items)")
    print(f"  Dataset:                 {item_key} ({item_dtype.__name__} IDs)")
    print(f"  Source:                  {metadata_path}")
    print(f"  Items total:             {n_total}")
    print(f"  Items with empty doc:    {n_empty}")
    print(f"  Items with >= 1 feature: {n_with_features}")
    print(f"  Items with 0 features:   {n_total - n_with_features}")
    print(f"  Vocabulary size:         {len(vocab)}")
    print(f"  Output:                  {output_path}")
    print(f"  Vocab:                   {vocab_path}")


# ── Per-folder mode ───────────────────────────────────────────────────────────

def _load_tsv_item_ids(dataset_dir: Path, item_dtype) -> set:
    """Union of item IDs (column index 1, no header) from train.tsv and test.tsv."""
    ids: set = set()
    for fname in ("train.tsv", "test.tsv"):
        path = dataset_dir / fname
        if path.exists():
            df = pd.read_csv(path, sep="\t", header=None, usecols=[1], dtype={1: item_dtype})
            ids.update(df[1].tolist())
    return ids


def _attr_file_item_ids(attr_path: Path, item_dtype) -> set:
    ids: set = set()
    with open(attr_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(item_dtype(line.split("\t")[0]))
    return ids


def build_item_attributes(
    metadata_path: Path,
    dataset_dir: Path,
    output_path: Path,
    item_key: str,
    item_dtype,
    text_cols: list[str],
    meta_dtype: dict,
    min_df: int = MIN_DF,
    max_features: int = MAX_FEATURES,
) -> None:
    """Build item_attributes.tsv filtered to items in train.tsv + test.tsv of dataset_dir."""
    dataset_dir = Path(dataset_dir)
    output_path = Path(output_path)

    tsv_ids = _load_tsv_item_ids(dataset_dir, item_dtype)
    if not tsv_ids:
        raise FileNotFoundError(
            f"Neither train.tsv nor test.tsv found in {dataset_dir}. "
            f"Run main.py first."
        )
    item_ids = sorted(tsv_ids)
    n_tsv_items = len(tsv_ids)

    # Optional consistency check against items.csv
    items_csv = dataset_dir / "items.csv"
    if items_csv.exists():
        csv_df = pd.read_csv(items_csv)
        csv_ids = set(csv_df.iloc[:, 0].apply(item_dtype).tolist())
        valid_only = csv_ids - tsv_ids
        if valid_only:
            print(f"  [INFO] items.csv has {len(valid_only)} valid-only items "
                  f"not in train/test — excluded (correct)")

    meta = _load_meta(metadata_path, meta_dtype)
    meta_filtered = meta[meta[item_key].apply(item_dtype).isin(tsv_ids)].copy().reset_index(drop=True)

    feature_map: dict = {}
    vocab: list[str] = []

    if meta_filtered.empty:
        print("  [WARN] No metadata found for any item; all items will have empty feature lists")
    else:
        meta_filtered = _build_doc_column(meta_filtered, text_cols)
        X, vocab = _vectorize(meta_filtered["doc"], min_df=min_df, max_features=max_features)
        feature_map = _build_feature_map(X, meta_filtered, item_key, item_dtype)

    _write_tsv(item_ids, feature_map, output_path)
    vocab_path = _write_vocab(vocab, output_path)

    # Validate
    written_ids = _attr_file_item_ids(output_path, item_dtype)
    if written_ids != tsv_ids:
        missing = tsv_ids - written_ids
        extra = written_ids - tsv_ids
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing (sample: {sorted(missing)[:5]})")
        if extra:
            parts.append(f"{len(extra)} extra (sample: {sorted(extra)[:5]})")
        raise RuntimeError(
            f"item_attributes.tsv item set does not match train+test: {'; '.join(parts)}"
        )
    print(f"  [OK] item_attributes.tsv covers all {len(tsv_ids)} train/test items")

    n_total = len(item_ids)
    n_with_meta = len(meta_filtered)
    n_missing_meta = n_total - n_with_meta
    n_empty_doc = int((meta_filtered["doc"].str.strip() == "").sum()) if not meta_filtered.empty else 0
    n_with_features = sum(1 for v in feature_map.values() if v)

    print(f"  Mode:                    per-folder")
    print(f"  Dataset dir:             {dataset_dir.name}")
    print(f"  Item key:                {item_key} ({item_dtype.__name__} IDs)")
    print(f"  Items in train+test:     {n_tsv_items}")
    print(f"  Items with metadata:     {n_with_meta}")
    print(f"  Items missing metadata:  {n_missing_meta}")
    print(f"  Items with empty doc:    {n_empty_doc}")
    print(f"  Items with >= 1 feature: {n_with_features}")
    print(f"  Items with 0 features:   {n_with_meta - n_with_features + n_missing_meta}")
    print(f"  Vocabulary size:         {len(vocab)}")
    print(f"  Output:                  {output_path}")
    print(f"  Vocab:                   {vocab_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    valid_datasets = ", ".join(_DATASET_PROFILES)
    p = argparse.ArgumentParser(
        description="Build Elliot VSM ItemAttributes file from item metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="hm", choices=list(_DATASET_PROFILES),
        help=f"Dataset profile: {valid_datasets} (default: hm)",
    )
    p.add_argument(
        "--metadata-path", required=True, type=Path,
        help="Path to metadata CSV (e.g. data/raw/metadata_hm.csv)",
    )
    # Global mode
    p.add_argument(
        "--global-output", default=None, type=Path, metavar="PATH",
        help=(
            "Build from ALL metadata items (no per-folder filter). "
            "Write to PATH. Example: data/processed/hm_item_attributes.tsv"
        ),
    )
    # Per-folder mode
    p.add_argument(
        "--dataset-dir", default=None, type=Path,
        help="Single dataset folder (per-folder mode)",
    )
    p.add_argument(
        "--output-path", default=None, type=Path,
        help="Output .tsv path for per-folder mode (default: <dataset-dir>/item_attributes.tsv)",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Per-folder mode: process all folders under data/processed/",
    )
    p.add_argument(
        "--filter", default="", metavar="TEXT",
        help="With --all: only process folders whose name contains TEXT",
    )
    # Shared options
    p.add_argument(
        "--min-df", type=int, default=MIN_DF,
        help=f"CountVectorizer min_df (default: {MIN_DF})",
    )
    p.add_argument(
        "--max-features", type=int, default=MAX_FEATURES,
        help=f"CountVectorizer max_features (default: {MAX_FEATURES})",
    )
    return p.parse_args()


def _has_data(folder: Path) -> bool:
    return (folder / "train.tsv").exists() or (folder / "test.tsv").exists()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata_path)

    if not metadata_path.exists():
        print(f"[ERROR] Metadata file not found: {metadata_path}")
        return

    profile = _DATASET_PROFILES[args.dataset]
    item_key = profile["item_key"]
    item_dtype = profile["item_dtype"]
    text_cols = profile["text_cols"]
    meta_dtype = profile["meta_dtype"]

    # ── Global mode ───────────────────────────────────────────────────────────
    if args.global_output is not None:
        output_path = Path(args.global_output)
        print(f"\n[BUILD GLOBAL] {output_path}")
        build_global_item_attributes(
            metadata_path=metadata_path,
            output_path=output_path,
            item_key=item_key,
            item_dtype=item_dtype,
            text_cols=text_cols,
            meta_dtype=meta_dtype,
            min_df=args.min_df,
            max_features=args.max_features,
        )
        return

    # ── Per-folder mode ───────────────────────────────────────────────────────
    if args.all:
        if not PROCESSED_DIR.exists():
            print(f"[ERROR] Processed dir not found: {PROCESSED_DIR}")
            return
        folders = sorted(p for p in PROCESSED_DIR.iterdir() if p.is_dir() and _has_data(p))
        if args.filter:
            folders = [p for p in folders if args.filter in p.name]
        if not folders:
            print("[WARN] No matching folders found.")
            return
        for folder in folders:
            output_path = folder / "item_attributes.tsv"
            print(f"\n[BUILD] {folder.name}")
            try:
                build_item_attributes(
                    metadata_path=metadata_path,
                    dataset_dir=folder,
                    output_path=output_path,
                    item_key=item_key,
                    item_dtype=item_dtype,
                    text_cols=text_cols,
                    meta_dtype=meta_dtype,
                    min_df=args.min_df,
                    max_features=args.max_features,
                )
            except Exception as e:
                print(f"  [ERROR] {e}")
        return

    if args.dataset_dir is None:
        print("[ERROR] Provide --global-output, --dataset-dir, or --all.")
        return

    dataset_dir = Path(args.dataset_dir)
    output_path = args.output_path or (dataset_dir / "item_attributes.tsv")
    print(f"\n[BUILD] {dataset_dir.name}")
    build_item_attributes(
        metadata_path=metadata_path,
        dataset_dir=dataset_dir,
        output_path=output_path,
        item_key=item_key,
        item_dtype=item_dtype,
        text_cols=text_cols,
        meta_dtype=meta_dtype,
        min_df=args.min_df,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
