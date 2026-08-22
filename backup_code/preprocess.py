import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import get_dataset_config
from src.io_utils import load_dataframe
from src.kcore import make_k_core
from src.dedup import deduplicate_user_item
from src.splitting import userwise_temporal_holdout_with_coverage
from src.thinning import (
    generate_random_thinning_levels,
    generate_head_item_cut_levels,
    generate_tail_item_cut_levels
)
from src.metrics import build_reference_stats, compute_sparsity_metrics
from src.dataset_folder import (
    save_dataset_folder,
    save_train_valid_test_folder,
)

# ── Default constants (edit here, or override via CLI args) ──────────────────
DATASET_NAME   = "amazon"   # "hm"
K_USER         = 12              # min interactions per user
K_ITEM         = 12         # min interactions per item
TEST_SIZE       = 0.1                           # pool → test
VALID_SIZE      = 0.1
KEEP_FRACS     = [0.9, 0.7, 0.5, 0.3, 0.1]
HEAD_KEEP_FRACS = [0.1]
TAIL_KEEP_FRACS = [0.1]
SEED           = 42
DEDUP_USER_ITEM = True            # True → keep last interaction per user-item pair

# # ── H&M preset ────────────────────────────────────────
# DATASET_NAME    = "hm"
# K_USER          = 35
# K_ITEM          = 25
# TEST_SIZE       = 0.1                           # pool → test
# VALID_SIZE      = 0.1                           # train_pool → valid
# KEEP_FRACS      = [0.9, 0.7, 0.5, 0.1, 0.05]   # random thinning levels
# HEAD_KEEP_FRACS = [0.1]
# TAIL_KEEP_FRACS = [0.1]
# SEED            = 42
# DEDUP_USER_ITEM = True


def _desired_holdout_sum(df: pd.DataFrame, user_col: str, holdout_size: float) -> int:
    """Sum of per-user desired holdout quotas: sum(min(ceil(n_u * size), n_u-1))."""
    sizes = df.groupby(user_col).size()
    desired = np.minimum(
        np.ceil(sizes * holdout_size).astype(int),
        (sizes - 1).clip(lower=0),
    )
    return int(desired.sum())


def parse_args():
    p = argparse.ArgumentParser(description="Run recsys preprocessing pipeline.")
    p.add_argument("--dataset", default=None,
                   help=f"Dataset name (default: {DATASET_NAME})")
    p.add_argument("--k", type=int, default=None,
                   help="Set both --k-user and --k-item to the same value (backward compat)")
    p.add_argument("--k-user", type=int, default=None, dest="k_user",
                   help=f"Min interactions per user for k-core (default: {K_USER})")
    p.add_argument("--k-item", type=int, default=None, dest="k_item",
                   help=f"Min interactions per item for k-core (default: {K_ITEM})")
    return p.parse_args()


def _format_sig_pct(x, sig=2):
    """Format a fraction (0–1) as a percentage string with `sig` significant figures.

    Examples (sig=2):
        0.0        -> "0"
        0.0000034  -> "0.00034"
        0.08075    -> "8.1"
        0.5        -> "50"
    """
    if pd.isna(x):
        return ""
    pct = x * 100
    if pct == 0:
        return "0"
    return "{:.{}g}".format(pct, sig)


def _log_df(label: str, df: pd.DataFrame, user_col: str, item_col: str) -> None:
    """Print n_interactions / n_users / n_items for a dataframe."""
    n_rows  = len(df)
    n_users = df[user_col].nunique() if n_rows > 0 else 0
    n_items = df[item_col].nunique() if n_rows > 0 else 0
    print(
        f"  {label}: interactions={n_rows:,}  users={n_users:,}  items={n_items:,}"
    )


def _dataframe_hash(df: pd.DataFrame, key_cols: list[str]) -> int:
    """Sort-stable hash of a DataFrame by key columns."""
    normalized = (
        df[key_cols]
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    return int(pd.util.hash_pandas_object(normalized, index=False).sum())


def _verify_split(
    label: str,
    thinned_pool: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    expected_test_hash: int,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    valid_ratio: float,
    report_rows: list,
) -> None:
    """Run consistency checks and append a row to report_rows."""
    key_cols  = [user_col, item_col]
    hash_cols = [user_col, item_col, timestamp_col]

    # 1. Row count
    assert len(train) + len(valid) == len(thinned_pool), (
        f"[{label}] ROW_COUNT: {len(train)}+{len(valid)}="
        f"{len(train)+len(valid)} != thinned_pool={len(thinned_pool)}"
    )

    # 2. User/item coverage in train
    pool_users = set(thinned_pool[user_col])
    pool_items = set(thinned_pool[item_col])
    train_users = set(train[user_col])
    train_items = set(train[item_col])
    missing_users = pool_users - train_users
    missing_items = pool_items - train_items
    user_cov_ok = len(missing_users) == 0
    item_cov_ok = len(missing_items) == 0
    if not user_cov_ok:
        print(f"  [WARN][{label}] train missing {len(missing_users):,} users from pool")
    if not item_cov_ok:
        print(f"  [WARN][{label}] train missing {len(missing_items):,} items from pool")

    # 3. Duplicates
    train_dups = int(train.duplicated(subset=key_cols).sum())
    valid_dups = int(valid.duplicated(subset=key_cols).sum())
    test_dups  = int(test.duplicated(subset=key_cols).sum())
    for name, n in [("train", train_dups), ("valid", valid_dups), ("test", test_dups)]:
        if n > 0:
            print(f"  [WARN][{label}] {name} has {n:,} duplicate (user,item) pairs")

    # 4. Pairwise overlap on (user, item) key
    train_set = set(map(tuple, train[key_cols].values))
    valid_set = set(map(tuple, valid[key_cols].values))
    test_set  = set(map(tuple, test[key_cols].values))
    tv_overlap = len(train_set & valid_set)
    tt_overlap = len(train_set & test_set)
    vt_overlap = len(valid_set & test_set)
    for pair, n in [("train&valid", tv_overlap), ("train&test", tt_overlap), ("valid&test", vt_overlap)]:
        if n > 0:
            print(f"  [WARN][{label}] {pair} overlap: {n:,} (user,item) pairs")

    # 5. Per-user quota check (replaces global ratio check)
    #    valid_count_u <= min(ceil(n_u * valid_ratio), n_u - 1) for every user
    input_counts = thinned_pool.groupby(user_col).size()
    valid_counts = (
        valid.groupby(user_col).size()
        .reindex(input_counts.index, fill_value=0)
    )
    desired_counts = np.minimum(
        np.ceil(input_counts * valid_ratio).astype(int),
        (input_counts - 1).clip(lower=0),
    )
    violations = valid_counts[valid_counts > desired_counts]
    quota_ok = violations.empty
    if not quota_ok:
        print(
            f"  [WARN][{label}] per-user quota exceeded for "
            f"{len(violations):,} users: {violations.to_dict()}"
        )

    desired_valid_sum = int(desired_counts.sum())
    actual_valid_rows = len(valid)
    valid_shortfall   = desired_valid_sum - actual_valid_rows
    actual_valid_ratio = actual_valid_rows / len(thinned_pool) if len(thinned_pool) > 0 else 0.0

    # 6. Test hash
    test_hash = _dataframe_hash(test, hash_cols)
    hash_ok = (test_hash == expected_test_hash)
    if not hash_ok:
        print(
            f"  [WARN][{label}] test hash mismatch! "
            f"expected={expected_test_hash}, got={test_hash}"
        )

    report_rows.append({
        "config":             label,
        "thinned_pool":       len(thinned_pool),
        "train":              len(train),
        "valid":              actual_valid_rows,
        "actual_valid_ratio": f"{actual_valid_ratio:.4f}",
        "desired_valid_sum":  desired_valid_sum,
        "valid_shortfall":    valid_shortfall,
        "quota_ok":           "OK" if quota_ok else "WARN",
        "test":               len(test),
        "test_hash_ok":       "OK" if hash_ok else "FAIL",
        "user_cov":           "OK" if user_cov_ok else "FAIL",
        "item_cov":           "OK" if item_cov_ok else "FAIL",
        "train_dups":         train_dups,
        "valid_dups":         valid_dups,
        "test_dups":          test_dups,
        "tv_overlap":         tv_overlap,
        "tt_overlap":         tt_overlap,
        "vt_overlap":         vt_overlap,
    })


def _assert_split(
    label: str,
    input_df: pd.DataFrame,
    train_part: pd.DataFrame,
    holdout_part: pd.DataFrame,
    user_col: str,
    item_col: str,
) -> None:
    """Explicit post-split assertions enforced in the pipeline (not just inside the split fn)."""
    assert len(train_part) + len(holdout_part) == len(input_df), (
        f"[{label}] Row count: {len(train_part)} + {len(holdout_part)} "
        f"!= input {len(input_df)}"
    )
    assert set(train_part[user_col]) == set(input_df[user_col]), (
        f"[{label}] Train user set != input user set. "
        f"Missing: {set(input_df[user_col]) - set(train_part[user_col])}"
    )
    assert set(train_part[item_col]) == set(input_df[item_col]), (
        f"[{label}] Train item set != input item set. "
        f"Missing: {set(input_df[item_col]) - set(train_part[item_col])}"
    )


def _format_report_df(rows) -> pd.DataFrame:
    """Format raw metric rows into the standardized summary DataFrame.

    Input rows contain raw metric values from compute_sparsity_metrics.
    Output has 9 columns with units/rounding applied:
        keep_frac, n_interactions_x1000, oss_pct, uss, iss,
        user_gini, item_gini, coldstart_user_pct, coldstart_item_pct
    """
    df = pd.DataFrame(rows)
    out = pd.DataFrame()
    out["keep_frac"]            = df["keep_frac"].round(2)
    out["n_interactions_x1000"] = (df["n_interactions"] / 1000).round(0).astype(int)
    out["oss_pct"]              = (df["oss"] * 100).round(3)
    out["uss"]                  = df["uss"].round(1)
    out["iss"]                  = df["iss"].round(1)
    out["user_gini"]            = df["user_gini"].round(2)
    out["item_gini"]            = df["item_gini"].round(2)
    out["coldstart_user_pct"]   = df["coldstart_user"].apply(_format_sig_pct)
    out["coldstart_item_pct"]   = df["coldstart_item"].apply(_format_sig_pct)
    return out


def _make_thinning_metadata(
    dataset_name, method, base_output, preprocessing_method,
    user_col, item_col, timestamp_col, k_user, k_item,
    keep_frac, reference_stats, train_metrics, test_base,
    desired_test_rows_sum, desired_valid_rows_sum, thin_train, thin_valid,
):
    return {
        "dataset_name": dataset_name,
        "method": method,
        "parent_dataset": str(base_output),
        "preprocessing_method": preprocessing_method,
        "user_col": user_col,
        "item_col": item_col,
        "timestamp_col": timestamp_col,
        "k_user": k_user,
        "k_item": k_item,
        "test_size": TEST_SIZE,
        "valid_size_from_train_pool": VALID_SIZE,
        "split_method": "userwise_temporal_holdout_with_coverage",
        "split_ratio_type": "per_user_ceil",
        "holdout_selection": "latest_temporal_suffix_per_user",
        "global_target_enforced": False,
        "shortfall_redistribution": False,
        "test_protection_policy": "recompute_protected_rows_before_test_split",
        "thinning_protection_policy": "protect_user_item_before_thinning",
        "valid_protection_policy": "recompute_protected_rows_before_valid_split",
        "protected_row_definition": "first_interaction_per_user_plus_first_interaction_per_uncovered_item",
        "holdout_window": "contiguous_suffix_after_last_protected_interaction_per_user",
        "dedup_user_item": DEDUP_USER_ITEM,
        "dedup_keep": "last" if DEDUP_USER_ITEM else None,
        "keep_frac": keep_frac,
        "reference_stats": reference_stats,
        "train_metrics": train_metrics,
        "desired_test_rows_sum": int(desired_test_rows_sum),
        "actual_test_rows": int(len(test_base)),
        "test_shortfall": int(desired_test_rows_sum - len(test_base)),
        "desired_valid_rows_sum": int(desired_valid_rows_sum),
        "actual_valid_rows": int(len(thin_valid)),
        "valid_shortfall": int(desired_valid_rows_sum - len(thin_valid)),
        "train_users": int(thin_train[user_col].nunique()),
        "train_items": int(thin_train[item_col].nunique()),
        "valid_users": int(thin_valid[user_col].nunique()),
        "valid_items": int(thin_valid[item_col].nunique()),
        "test_users":  int(test_base[user_col].nunique()),
        "test_items":  int(test_base[item_col].nunique()),
        "test_rows":   int(len(test_base)),
    }


def main():
    args = parse_args()

    dataset_name = args.dataset or DATASET_NAME

    # Resolve k values: --k sets both; --k-user/--k-item override individually
    if args.k is not None:
        k_user = args.k
        k_item = args.k
    else:
        k_user = args.k_user if args.k_user is not None else K_USER
        k_item = args.k_item if args.k_item is not None else K_ITEM

    # load dataset
    cfg = get_dataset_config(dataset_name)
    input_path    = cfg["input_path"]
    user_col      = cfg["user_col"]
    item_col      = cfg["item_col"]
    timestamp_col = cfg["timestamp_col"]

    output_root = Path("data/processed")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Step 1: Load raw data")
    df = load_dataframe(input_path)
    print(f"  Raw shape: {df.shape}")

    if DEDUP_USER_ITEM:
        print("Step 2: Deduplicate user-item pairs")
        processed_df = deduplicate_user_item(
            df=df,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            keep='last'
        )
        preprocessing_method = "dedup_user_item"
        version_name = "dedup"
        print(f"  After dedup shape: {processed_df.shape}")
        print(f"  Removed duplicate rows: {len(df) - len(processed_df)}")
    else:
        print("Step 2: Keep duplicate user-item interactions")
        processed_df = df.copy()
        preprocessing_method = "keep_duplicate_interactions"
        version_name = "transaction"
        print(f"  After preprocessing shape: {processed_df.shape}")

    print(f"Step 3: Apply k-core  (k_user={k_user}, k_item={k_item})")
    dense_df = make_k_core(
        df=processed_df,
        user_col=user_col,
        item_col=item_col,
        k_user=k_user,
        k_item=k_item,
        verbose=True,
    )
    _log_df("dense_df", dense_df, user_col, item_col)

    dense_reference_stats = build_reference_stats(
        dense_df, user_col=user_col, item_col=item_col
    )
    dense_metrics = compute_sparsity_metrics(
        dense_df, user_col=user_col, item_col=item_col, reference_stats=dense_reference_stats
    )

    dense_output = output_root / f"{dataset_name}_{version_name}"
    dense_metadata = {
        "dataset_name": dataset_name,
        "method": f"{preprocessing_method}_kcore",
        "user_col": user_col,
        "item_col": item_col,
        "timestamp_col": timestamp_col,
        "k_user": k_user,
        "k_item": k_item,
        "dedup_user_item": DEDUP_USER_ITEM,
        "dedup_keep": "last" if DEDUP_USER_ITEM else None,
        "reference_stats": dense_reference_stats,
        "metrics": dense_metrics,
    }

    save_dataset_folder(
        df=dense_df,
        output_dir=dense_output,
        user_col=user_col,
        item_col=item_col,
        metadata=dense_metadata,
    )

    # ── Step 4: dense_df → train_pool_base / test_base ──────────────────────
    desired_test_rows_sum = _desired_holdout_sum(dense_df, user_col, TEST_SIZE)
    print(
        f"Step 4: Per-user temporal holdout -> train_pool / test  "
        f"(test_size={TEST_SIZE}, desired_per_user_sum={desired_test_rows_sum:,})"
    )
    train_pool_base, test_base = userwise_temporal_holdout_with_coverage(
        df=dense_df,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        holdout_size=TEST_SIZE,
        split_name="test",
    )
    _log_df("train_pool_base", train_pool_base, user_col, item_col)
    _log_df("test_base      ", test_base,       user_col, item_col)
    print(
        f"  actual_test_rows={len(test_base):,} | "
        f"desired_sum={desired_test_rows_sum:,} | "
        f"shortfall={desired_test_rows_sum - len(test_base):,}"
    )
    _assert_split("step4_test", dense_df, train_pool_base, test_base, user_col, item_col)

    # Compute test hash once — all configurations must produce the same test set
    _test_hash_cols = [user_col, item_col, timestamp_col]
    test_hash = _dataframe_hash(test_base, _test_hash_cols)
    print(f"  test_base hash: {test_hash}")
    split_report_rows: list[dict] = []

    # ── Step 4b: train_pool_base → train_base / valid_base ──────────────────
    desired_valid_base_sum = _desired_holdout_sum(train_pool_base, user_col, VALID_SIZE)
    print(
        f"Step 4b: Per-user temporal holdout -> train / valid  "
        f"(valid_size={VALID_SIZE}, desired_per_user_sum={desired_valid_base_sum:,})"
    )
    train_base, valid_base = userwise_temporal_holdout_with_coverage(
        df=train_pool_base,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        holdout_size=VALID_SIZE,
        split_name="valid_base",
    )
    _log_df("train_base", train_base, user_col, item_col)
    _log_df("valid_base", valid_base, user_col, item_col)
    _assert_split("step4b_valid_base", train_pool_base, train_base, valid_base, user_col, item_col)
    _verify_split(
        f"{dataset_name}_base", train_pool_base,
        train_base, valid_base, test_base,
        test_hash, user_col, item_col, timestamp_col, VALID_SIZE, split_report_rows,
    )

    # Reference population for all summary reports = final base train set
    reference_stats = build_reference_stats(train_base, user_col=user_col, item_col=item_col)
    train_base_metrics = compute_sparsity_metrics(
        train_base, user_col=user_col, item_col=item_col, reference_stats=reference_stats
    )

    # ── Sanity check ────────────────────────────────────────────────────────
    _ref_n_u = reference_stats["base_n_users"]
    _ref_n_i = reference_stats["base_n_items"]
    _n_int   = train_base_metrics["n_interactions"]
    _oss_expected = _n_int / (_ref_n_u * _ref_n_i) * 100
    _oss_actual   = train_base_metrics["oss"] * 100
    assert abs(_oss_expected - _oss_actual) < 1e-9, (
        f"OSS sanity fail: expected {_oss_expected}, got {_oss_actual}"
    )
    print(
        f"  [sanity] keep=1.0 | interactions={_n_int:,} | "
        f"n_users={_ref_n_u:,} | n_items={_ref_n_i:,} | "
        f"oss={_oss_actual:.4f}% | "
        f"coldstart_user={_format_sig_pct(train_base_metrics['coldstart_user'])}% | "
        f"coldstart_item={_format_sig_pct(train_base_metrics['coldstart_item'])}%"
    )

    base_output = output_root / f"{dataset_name}_{version_name}_base_split"
    base_metadata = {
        "dataset_name": dataset_name,
        "method": "userwise_temporal_holdout_with_coverage",
        "parent_dataset": str(dense_output),
        "preprocessing_method": preprocessing_method,
        "user_col": user_col,
        "item_col": item_col,
        "timestamp_col": timestamp_col,
        "k_user": k_user,
        "k_item": k_item,
        "test_size": TEST_SIZE,
        "valid_size_from_train_pool": VALID_SIZE,
        "split_method": "userwise_temporal_holdout_with_coverage",
        "split_ratio_type": "per_user_ceil",
        "holdout_selection": "latest_temporal_suffix_per_user",
        "global_target_enforced": False,
        "shortfall_redistribution": False,
        "test_protection_policy": "recompute_protected_rows_before_test_split",
        "thinning_protection_policy": "protect_user_item_before_thinning",
        "valid_protection_policy": "recompute_protected_rows_before_valid_split",
        "protected_row_definition": "first_interaction_per_user_plus_first_interaction_per_uncovered_item",
        "holdout_window": "contiguous_suffix_after_last_protected_interaction_per_user",
        "dedup_user_item": DEDUP_USER_ITEM,
        "dedup_keep": "last" if DEDUP_USER_ITEM else None,
        "reference_stats": reference_stats,
        "train_metrics": train_base_metrics,
        "desired_test_rows_sum": int(desired_test_rows_sum),
        "actual_test_rows": int(len(test_base)),
        "test_shortfall": int(desired_test_rows_sum - len(test_base)),
        "desired_valid_rows_sum": int(desired_valid_base_sum),
        "actual_valid_rows": int(len(valid_base)),
        "valid_shortfall": int(desired_valid_base_sum - len(valid_base)),
        "train_users": int(train_base[user_col].nunique()),
        "train_items": int(train_base[item_col].nunique()),
        "valid_users": int(valid_base[user_col].nunique()),
        "valid_items": int(valid_base[item_col].nunique()),
        "test_users":  int(test_base[user_col].nunique()),
        "test_items":  int(test_base[item_col].nunique()),
        "test_rows":   int(len(test_base)),
    }

    save_train_valid_test_folder(
        train_df=train_base,
        valid_df=valid_base,
        test_df=test_base,
        output_dir=base_output,
        user_col=user_col,
        item_col=item_col,
        metadata=base_metadata,
    )

    # ── Step 5: Random thinning on train_pool_base → thin_train_pool → train/valid ──
    print("Step 5: Random thinning")
    random_pool_outputs = generate_random_thinning_levels(
        df=train_pool_base,
        keep_fracs=KEEP_FRACS,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        seed=SEED,
    )

    random_report_rows = [{"keep_frac": 1.0, **train_base_metrics}]

    for level_name, thin_train_pool in random_pool_outputs.items():
        _log_df(f"  random {level_name} thin_train_pool", thin_train_pool, user_col, item_col)
        desired_valid_rows_sum = _desired_holdout_sum(thin_train_pool, user_col, VALID_SIZE)
        thin_train, thin_valid = userwise_temporal_holdout_with_coverage(
            df=thin_train_pool,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            holdout_size=VALID_SIZE,
            split_name=f"valid_{dataset_name}_random_{level_name}",
        )
        _log_df(f"  random {level_name} train", thin_train, user_col, item_col)
        _log_df(f"  random {level_name} valid", thin_valid, user_col, item_col)
        _assert_split(f"random_{level_name}_valid", thin_train_pool, thin_train, thin_valid, user_col, item_col)
        _verify_split(
            f"{dataset_name}_random_{level_name}",
            thin_train_pool, thin_train, thin_valid, test_base,
            test_hash, user_col, item_col, timestamp_col, VALID_SIZE, split_report_rows,
        )
        thin_metrics = compute_sparsity_metrics(
            thin_train, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )
        thin_output = output_root / f"{dataset_name}_random_{level_name}"
        keep_frac = len(thin_train_pool) / len(train_pool_base)
        thin_metadata = _make_thinning_metadata(
            dataset_name=dataset_name, method="random_thinning_train_only",
            base_output=base_output, preprocessing_method=preprocessing_method,
            user_col=user_col, item_col=item_col, timestamp_col=timestamp_col,
            k_user=k_user, k_item=k_item, keep_frac=keep_frac,
            reference_stats=reference_stats, train_metrics=thin_metrics, test_base=test_base,
            desired_test_rows_sum=desired_test_rows_sum,
            desired_valid_rows_sum=desired_valid_rows_sum,
            thin_train=thin_train, thin_valid=thin_valid,
        )
        save_train_valid_test_folder(
            train_df=thin_train, valid_df=thin_valid, test_df=test_base,
            output_dir=thin_output, user_col=user_col, item_col=item_col,
            metadata=thin_metadata,
        )
        random_report_rows.append({"keep_frac": keep_frac, **thin_metrics})

    # ── Step 6: Head-item thinning on train_pool_base ───────────────────────
    print("Step 6: Head-item thinning")
    head_pool_outputs = generate_head_item_cut_levels(
        df=train_pool_base,
        keep_fracs=HEAD_KEEP_FRACS,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        seed=SEED,
    )

    head_report_rows = [{"keep_frac": 1.0, **train_base_metrics}]

    for level_name, thin_train_pool in head_pool_outputs.items():
        _log_df(f"  head {level_name} thin_train_pool", thin_train_pool, user_col, item_col)
        desired_valid_rows_sum = _desired_holdout_sum(thin_train_pool, user_col, VALID_SIZE)
        thin_train, thin_valid = userwise_temporal_holdout_with_coverage(
            df=thin_train_pool,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            holdout_size=VALID_SIZE,
            split_name=f"valid_{dataset_name}_head_{level_name}",
        )
        _log_df(f"  head {level_name} train", thin_train, user_col, item_col)
        _log_df(f"  head {level_name} valid", thin_valid, user_col, item_col)
        _assert_split(f"head_{level_name}_valid", thin_train_pool, thin_train, thin_valid, user_col, item_col)
        _verify_split(
            f"{dataset_name}_head_{level_name}",
            thin_train_pool, thin_train, thin_valid, test_base,
            test_hash, user_col, item_col, timestamp_col, VALID_SIZE, split_report_rows,
        )
        thin_metrics = compute_sparsity_metrics(
            thin_train, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )
        thin_output = output_root / f"{dataset_name}_head_{level_name}"
        keep_frac = len(thin_train_pool) / len(train_pool_base)
        thin_metadata = _make_thinning_metadata(
            dataset_name=dataset_name, method="head_item_cut_train_only",
            base_output=base_output, preprocessing_method=preprocessing_method,
            user_col=user_col, item_col=item_col, timestamp_col=timestamp_col,
            k_user=k_user, k_item=k_item, keep_frac=keep_frac,
            reference_stats=reference_stats, train_metrics=thin_metrics, test_base=test_base,
            desired_test_rows_sum=desired_test_rows_sum,
            desired_valid_rows_sum=desired_valid_rows_sum,
            thin_train=thin_train, thin_valid=thin_valid,
        )
        save_train_valid_test_folder(
            train_df=thin_train, valid_df=thin_valid, test_df=test_base,
            output_dir=thin_output, user_col=user_col, item_col=item_col,
            metadata=thin_metadata,
        )
        head_report_rows.append({"keep_frac": keep_frac, **thin_metrics})

    # ── Step 7: Tail-item thinning on train_pool_base ───────────────────────
    print("Step 7: Tail-item thinning")
    tail_pool_outputs = generate_tail_item_cut_levels(
        df=train_pool_base,
        keep_fracs=TAIL_KEEP_FRACS,
        user_col=user_col,
        item_col=item_col,
        timestamp_col=timestamp_col,
        seed=SEED,
    )

    tail_report_rows = [{"keep_frac": 1.0, **train_base_metrics}]

    for level_name, thin_train_pool in tail_pool_outputs.items():
        _log_df(f"  tail {level_name} thin_train_pool", thin_train_pool, user_col, item_col)
        desired_valid_rows_sum = _desired_holdout_sum(thin_train_pool, user_col, VALID_SIZE)
        thin_train, thin_valid = userwise_temporal_holdout_with_coverage(
            df=thin_train_pool,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            holdout_size=VALID_SIZE,
            split_name=f"valid_{dataset_name}_tail_{level_name}",
        )
        _log_df(f"  tail {level_name} train", thin_train, user_col, item_col)
        _log_df(f"  tail {level_name} valid", thin_valid, user_col, item_col)
        _assert_split(f"tail_{level_name}_valid", thin_train_pool, thin_train, thin_valid, user_col, item_col)
        _verify_split(
            f"{dataset_name}_tail_{level_name}",
            thin_train_pool, thin_train, thin_valid, test_base,
            test_hash, user_col, item_col, timestamp_col, VALID_SIZE, split_report_rows,
        )
        thin_metrics = compute_sparsity_metrics(
            thin_train, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )
        thin_output = output_root / f"{dataset_name}_tail_{level_name}"
        keep_frac = len(thin_train_pool) / len(train_pool_base)
        thin_metadata = _make_thinning_metadata(
            dataset_name=dataset_name, method="tail_item_cut_train_only",
            base_output=base_output, preprocessing_method=preprocessing_method,
            user_col=user_col, item_col=item_col, timestamp_col=timestamp_col,
            k_user=k_user, k_item=k_item, keep_frac=keep_frac,
            reference_stats=reference_stats, train_metrics=thin_metrics, test_base=test_base,
            desired_test_rows_sum=desired_test_rows_sum,
            desired_valid_rows_sum=desired_valid_rows_sum,
            thin_train=thin_train, thin_valid=thin_valid,
        )
        save_train_valid_test_folder(
            train_df=thin_train, valid_df=thin_valid, test_df=test_base,
            output_dir=thin_output, user_col=user_col, item_col=item_col,
            metadata=thin_metadata,
        )
        tail_report_rows.append({"keep_frac": keep_frac, **thin_metrics})

    Path("data/reports").mkdir(parents=True, exist_ok=True)

    _format_report_df(random_report_rows).to_csv(
        f"data/reports/{dataset_name}_random_summary.csv", index=False
    )
    _format_report_df(head_report_rows).to_csv(
        f"data/reports/{dataset_name}_head_summary.csv", index=False
    )
    _format_report_df(tail_report_rows).to_csv(
        f"data/reports/{dataset_name}_tail_summary.csv", index=False
    )

    print("\n-- Split verification report -------------------------------------------")
    report_df = pd.DataFrame(split_report_rows, columns=[
        "config", "thinned_pool", "train", "valid",
        "actual_valid_ratio", "desired_valid_sum", "valid_shortfall", "quota_ok",
        "test", "test_hash_ok", "user_cov", "item_cov",
        "train_dups", "valid_dups", "test_dups",
        "tv_overlap", "tt_overlap", "vt_overlap",
    ])
    print(report_df.to_string(index=False))
    print("------------------------------------------------------------------------")

    print("Done.")


if __name__ == "__main__":
    main()
