"""
Convert pipeline train/valid/test CSVs to RecBole Atomic Files (.inter).

Each atomic file has exactly four columns:
    user_id:token  item_id:token  timestamp:float  item_id_list:token_seq

Source : data/processed/<dataset>/{train,valid,test}.csv
Output : data/recbole/<dataset>/<dataset>.{train,valid,test}.inter

Row count is preserved exactly — no dedup, no filter, no sort.

Usage:
    python scripts/recbole_prepare_data.py --dataset hm_random_keep0.1 --overwrite
    python scripts/recbole_prepare_data.py --all --overwrite
"""

import argparse
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

RECBOLE_HEADER = (
    "user_id:token\titem_id:token\ttimestamp:float\titem_id_list:token_seq"
)

USER_ALIASES = ["user_id", "user", "uid", "userid", "customer_id"]
ITEM_ALIASES = ["item_id", "item", "iid", "itemid", "parent_asin", "article_id", "asin"]
TS_ALIASES   = ["timestamp", "time", "ts", "t_dat", "event_time"]


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _detect_columns(df: pd.DataFrame, path: Path) -> tuple[str, str, str]:
    """Return (user_col, item_col, ts_col) matched from df.columns.

    Raises if any role matches zero or more-than-one column.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    def _find(aliases: list[str], role: str) -> str:
        found = [cols_lower[a] for a in aliases if a in cols_lower]
        if len(found) == 0:
            raise ValueError(
                f"{role} column not found in {path}\n"
                f"  Columns present: {list(df.columns)}\n"
                f"  Expected one of: {aliases}"
            )
        if len(found) > 1:
            raise ValueError(
                f"Multiple {role} columns found in {path}: {found}\n"
                f"  Columns present: {list(df.columns)}"
            )
        return found[0]

    return _find(USER_ALIASES, "user"), _find(ITEM_ALIASES, "item"), _find(TS_ALIASES, "timestamp")


# ---------------------------------------------------------------------------
# Timestamp conversion
# ---------------------------------------------------------------------------

def _scale_to_unix_seconds(raw: pd.Series, path: Path) -> pd.Series:
    """Scale a numeric Series to Unix seconds based on median magnitude.

    Thresholds (powers of 10 between adjacent units):
        < 1e11  → already seconds  (covers up to year ~5138 in seconds)
        < 1e14  → milliseconds     ÷ 1e3
        < 1e17  → microseconds     ÷ 1e6
        else    → nanoseconds      ÷ 1e9
    """
    median = float(raw.abs().median())
    if median < 1e11:
        result = raw.astype(float)
    elif median < 1e14:
        result = (raw / 1e3).astype(float)
    elif median < 1e17:
        result = (raw / 1e6).astype(float)
    else:
        result = (raw / 1e9).astype(float)

    if result.median() < 100_000_000:
        raise ValueError(
            f"Timestamp values are implausibly small for Unix seconds in {path}\n"
            f"  median={result.median():.3e} — possible unit-conversion error.\n"
            f"  Raw median={median:.3e}, raw sample={raw.head(3).tolist()}"
        )
    return result


def _to_float_timestamp(series: pd.Series, path: Path) -> pd.Series:
    """Convert a Series to float Unix seconds.

    Flow:
      1. Null check on source.
      2. If all values parse as numeric: apply magnitude-based unit detection.
      3. Otherwise parse as datetime string (UTC), get nanosecond int64,
         then apply magnitude-based unit detection on that value.
    Raises on nulls or unparseable values with sample diagnostics.
    """
    null_count = series.isna().sum()
    if null_count > 0:
        raise ValueError(
            f"Timestamp column has {null_count} null values in {path}\n"
            f"  Sample null rows: {list(series[series.isna()].index[:5])}"
        )

    # Numeric path — do NOT pass through pd.to_datetime
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return _scale_to_unix_seconds(numeric, path)

    # Datetime string path
    try:
        parsed = pd.to_datetime(series, utc=True, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(series, utc=True, errors="coerce")

    bad_mask = parsed.isna()
    if bad_mask.any():
        bad_vals = series[bad_mask].head(5).tolist()
        raise ValueError(
            f"Timestamp not parseable in {path}\n"
            f"  {bad_mask.sum()} invalid values, e.g.: {bad_vals}"
        )

    # Get nanosecond int64, then scale to seconds
    ns = parsed.astype("int64").astype(float)
    return _scale_to_unix_seconds(ns, path)


# ---------------------------------------------------------------------------
# Load / write / validate
# ---------------------------------------------------------------------------

def _load_split(path: Path) -> tuple[pd.DataFrame, int]:
    """Load CSV; return the RecBole interaction fields and source row count.

    n_source_rows == len(df) always — no rows are dropped.
    """
    df_raw = pd.read_csv(path, dtype=str)
    n_input = len(df_raw)

    user_col, item_col, ts_col = _detect_columns(df_raw, path)

    # Null check for user / item
    for col, role in [(user_col, "user"), (item_col, "item")]:
        nulls = df_raw[col].isna().sum()
        if nulls > 0:
            raise ValueError(f"{role} column has {nulls} null values in {path}")

    if "item_id_list" not in df_raw.columns:
        raise ValueError(
            f"History column 'item_id_list' not found in {path}\n"
            f"  Columns present: {list(df_raw.columns)}"
        )

    ts_float = _to_float_timestamp(df_raw[ts_col], path)
    item_id_list = df_raw["item_id_list"].fillna("")

    out = pd.DataFrame({
        "user_id":      df_raw[user_col].values,
        "item_id":      df_raw[item_col].values,
        "timestamp":    ts_float.values,
        "item_id_list": item_id_list.values,
    })

    if len(out) != n_input:
        raise RuntimeError(
            f"Row count changed during load of {path}: "
            f"source={n_input}, loaded={len(out)}"
        )
    return out, n_input


def _write_inter(df: pd.DataFrame, out: Path) -> None:
    with open(out, "w", encoding="utf-8") as f:
        f.write(RECBOLE_HEADER + "\n")
        df.to_csv(f, sep="\t", index=False, header=False, lineterminator="\n")


def _validate_inter(path: Path, expected_rows: int) -> dict:
    """Read back written file and assert correctness. Returns stats dict."""
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    actual_cols = list(df.columns)
    expected_cols = [
        "user_id:token",
        "item_id:token",
        "timestamp:float",
        "item_id_list:token_seq",
    ]

    if actual_cols != expected_cols:
        raise AssertionError(
            f"Header mismatch in {path}\n"
            f"  Expected: {expected_cols}\n"
            f"  Got:      {actual_cols}"
        )
    if len(df) != expected_rows:
        raise AssertionError(
            f"Row count mismatch in {path}: expected {expected_rows}, got {len(df)}"
        )
    if df["user_id:token"].isna().any():
        raise AssertionError(f"Null user_id found in {path}")
    if df["item_id:token"].isna().any():
        raise AssertionError(f"Null item_id found in {path}")
    if df["timestamp:float"].isna().any():
        raise AssertionError(f"Null timestamp found in {path}")
    if df["item_id_list:token_seq"].isna().any():
        raise AssertionError(f"Null item_id_list found in {path}")

    ts = pd.to_numeric(df["timestamp:float"], errors="coerce")
    if ts.isna().any():
        raise AssertionError(f"Non-numeric timestamp found in {path}")

    return {"rows": len(df), "ts_min": float(ts.min()), "ts_max": float(ts.max())}


# ---------------------------------------------------------------------------
# Dataset conversion
# ---------------------------------------------------------------------------

def _convert_dataset(
    dataset: str, processed_dir: Path, output_dir: Path, overwrite: bool
) -> bool:
    src_dir = processed_dir / dataset
    out_dir = output_dir / dataset

    if not src_dir.is_dir():
        print(f"  [SKIP] {dataset}: not a directory")
        return False

    splits = [
        (src_dir / "train.csv", out_dir / f"{dataset}.train.inter", "train"),
        (src_dir / "valid.csv", out_dir / f"{dataset}.valid.inter", "valid"),
        (src_dir / "test.csv",  out_dir / f"{dataset}.test.inter",  "test"),
    ]

    # All source files must exist before writing anything
    for src, _, split_name in splits:
        if not src.exists():
            print(f"  [SKIP] {dataset}: {src.name} not found")
            return False

    # If all outputs already exist and --overwrite not set, skip the whole dataset
    if all(dst.exists() for _, dst, _ in splits) and not overwrite:
        print(f"  [SKIP] {dataset}: all inter files exist (use --overwrite)")
        return False

    # Load and validate all splits before writing (fail fast, no partial output)
    loaded: dict[str, tuple[Path, Path, pd.DataFrame, int]] = {}
    for src, dst, split_name in splits:
        df, n_input = _load_split(src)
        loaded[split_name] = (src, dst, df, n_input)

    # Write + post-write validation
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, (src, dst, df, n_input) in loaded.items():
        _write_inter(df, dst)
        stats = _validate_inter(dst, expected_rows=n_input)
        print(
            f"  [OK] dataset={dataset}  split={split_name}\n"
            f"       source_rows={n_input:,}  output_rows={stats['rows']:,}\n"
            f"       timestamp_min={stats['ts_min']}  timestamp_max={stats['ts_max']}\n"
            f"       output={dst}"
        )

    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Convert pipeline CSVs to RecBole atomic files (3-column, with timestamp)"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset variant name")
    group.add_argument("--all",     action="store_true",
                       help="Convert all datasets in processed_dir")
    p.add_argument("--processed_dir", default="data/processed")
    p.add_argument("--output_dir",    default="data/recbole")
    p.add_argument("--overwrite",     action="store_true")
    args = p.parse_args()

    processed_dir = _PROJECT_ROOT / args.processed_dir
    output_dir    = _PROJECT_ROOT / args.output_dir

    if args.dataset:
        datasets = [args.dataset]
    else:
        if not processed_dir.exists():
            raise SystemExit(f"ERROR: {processed_dir} does not exist.")
        datasets = sorted(d.name for d in processed_dir.iterdir() if d.is_dir())
        if not datasets:
            raise SystemExit(f"No subdirectories found in {processed_dir}")

    success = skip = error = 0
    for dataset in datasets:
        print(f"\n[{dataset}]")
        try:
            ok = _convert_dataset(dataset, processed_dir, output_dir, args.overwrite)
            if ok:
                success += 1
            else:
                skip += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            error += 1

    if args.all or len(datasets) > 1:
        print(f"\n--- Summary: {success} success, {skip} skip, {error} error ---")


if __name__ == "__main__":
    main()
