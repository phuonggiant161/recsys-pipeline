import argparse
from pathlib import Path
import pandas as pd

from src.config import get_dataset_config
from src.io_utils import load_dataframe
from src.kcore import make_k_core
from src.preprocessing import deduplicate_user_item
from src.splitting import userwise_temporal_split
from src.thinning import (
    generate_random_thinning_levels,
    generate_head_item_cut_levels,
    generate_tail_item_cut_levels
)
from src.metrics import build_reference_stats, compute_sparsity_metrics
from src.dataset_folder import save_dataset_folder, save_train_test_folder

# ── Default constants (edit here, or override via CLI args) ──────────────────
DATASET_NAME   = "baby_product"   # "hm"
K_USER         = 5              # min interactions per user
K_ITEM         = 15             # min interactions per item
TEST_SIZE      = 0.2
KEEP_FRACS     = [0.9, 0.75, 0.6, 0.45, 0.3]
HEAD_KEEP_FRACS = [0.5]                         # head-item thinning: fixed at 0.5
TAIL_KEEP_FRACS = [0.5] 
SEED           = 42
DEDUP_USER_ITEM = True            # True → keep last interaction per user-item pair

# ── H&M preset (uncomment to switch) ────────────────────────────────────────
# DATASET_NAME   = "hm"
# K_USER         = 15
# K_ITEM         = 30
# TEST_SIZE      = 0.2
# KEEP_FRACS      = [0.9, 0.7, 0.5, 0.3, 0.1]   # random thinning levels
# HEAD_KEEP_FRACS = [0.5]                         # head-item thinning: fixed at 0.5
# TAIL_KEEP_FRACS = [0.5]                         # tail-item thinning: fixed at 0.5
# SEED            = 42
# DEDUP_USER_ITEM = True


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

    reference_stats = build_reference_stats(
        dense_df, user_col=user_col, item_col=item_col
    )
    dense_metrics = compute_sparsity_metrics(
        dense_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
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
        "reference_stats": reference_stats,
        "metrics": dense_metrics,
    }

    save_dataset_folder(
        df=dense_df,
        output_dir=dense_output,
        user_col=user_col,
        item_col=item_col,
        metadata=dense_metadata,
    )

    print("Step 4: User-wise temporal train/test split")
    train_base, test_base = userwise_temporal_split(
        df=dense_df,
        user_col=user_col,
        timestamp_col=timestamp_col,
        test_size=TEST_SIZE,
        min_train_interactions=1,
        min_test_interactions=1
    )
    print(f"  Train base shape: {train_base.shape}")
    print(f"  Test base shape:  {test_base.shape}")

    train_base_metrics = compute_sparsity_metrics(
        train_base, user_col=user_col, item_col=item_col, reference_stats=reference_stats
    )

    base_output = output_root / f"{dataset_name}_{version_name}_base_split"
    base_metadata = {
        "dataset_name": dataset_name,
        "method": "userwise_temporal_split",
        "parent_dataset": str(dense_output),
        "preprocessing_method": preprocessing_method,
        "user_col": user_col,
        "item_col": item_col,
        "timestamp_col": timestamp_col,
        "k_user": k_user,
        "k_item": k_item,
        "test_size": TEST_SIZE,
        "split_method": "userwise_temporal_split",
        "dedup_user_item": DEDUP_USER_ITEM,
        "dedup_keep": "last" if DEDUP_USER_ITEM else None,
        "reference_stats": reference_stats,
        "train_metrics": train_base_metrics,
        "test_rows": int(len(test_base)),
        "test_users": int(test_base[user_col].nunique()),
        "test_items": int(test_base[item_col].nunique()),
    }

    save_train_test_folder(
        train_df=train_base,
        test_df=test_base,
        output_dir=base_output,
        user_col=user_col,
        item_col=item_col,
        metadata=base_metadata,
    )

    print("Step 5: Random thinning")
    thinning_outputs = generate_random_thinning_levels(
        df=train_base,
        keep_fracs=KEEP_FRACS,
        seed=SEED,
        user_col=user_col,
        timestamp_col=timestamp_col,
        protect_first_interaction=True
    )

    random_report_rows = [{"dataset": "base_train", "keep_frac": 1.0, **train_base_metrics}]

    for level_name, thin_train_df in thinning_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_train_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )
        thin_output = output_root / f"{dataset_name}_random_{level_name}"
        keep_frac = len(thin_train_df) / len(train_base)
        thin_metadata = {
            "dataset_name": dataset_name,
            "method": "random_thinning_train_only",
            "parent_dataset": str(base_output),
            "preprocessing_method": preprocessing_method,
            "user_col": user_col,
            "item_col": item_col,
            "timestamp_col": timestamp_col,
            "k_user": k_user,
            "k_item": k_item,
            "test_size": TEST_SIZE,
            "split_method": "userwise_temporal_split",
            "dedup_user_item": DEDUP_USER_ITEM,
            "dedup_keep": "last" if DEDUP_USER_ITEM else None,
            "keep_frac": keep_frac,
            "reference_stats": reference_stats,
            "train_metrics": thin_metrics,
            "test_policy": "fixed_test_from_base_split",
            "test_rows": int(len(test_base)),
        }
        save_train_test_folder(
            train_df=thin_train_df, test_df=test_base,
            output_dir=thin_output, user_col=user_col, item_col=item_col,
            metadata=thin_metadata,
        )
        random_report_rows.append({"dataset": level_name, "keep_frac": keep_frac, **thin_metrics})

    print("Step 6: Head-item thinning")
    head_outputs = generate_head_item_cut_levels(
        df=train_base,
        item_col=item_col,
        keep_fracs=HEAD_KEEP_FRACS,
        seed=SEED,
        user_col=user_col,
        timestamp_col=timestamp_col,
        protect_first_interaction=True
    )

    head_report_rows = [{"dataset": "base_train", "keep_frac": 1.0, **train_base_metrics}]

    for level_name, thin_train_df in head_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_train_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )
        thin_output = output_root / f"{dataset_name}_head_{level_name}"
        keep_frac = len(thin_train_df) / len(train_base)
        thin_metadata = {
            "dataset_name": dataset_name,
            "method": "head_item_cut_train_only",
            "parent_dataset": str(base_output),
            "preprocessing_method": preprocessing_method,
            "user_col": user_col,
            "item_col": item_col,
            "timestamp_col": timestamp_col,
            "k_user": k_user,
            "k_item": k_item,
            "test_size": TEST_SIZE,
            "split_method": "userwise_temporal_split",
            "dedup_user_item": DEDUP_USER_ITEM,
            "dedup_keep": "last" if DEDUP_USER_ITEM else None,
            "keep_frac": keep_frac,
            "reference_stats": reference_stats,
            "train_metrics": thin_metrics,
            "test_policy": "fixed_test_from_base_split",
            "test_rows": int(len(test_base)),
        }
        save_train_test_folder(
            train_df=thin_train_df, test_df=test_base,
            output_dir=thin_output, user_col=user_col, item_col=item_col,
            metadata=thin_metadata,
        )
        head_report_rows.append({"dataset": level_name, "keep_frac": keep_frac, **thin_metrics})

    print("Step 7: Tail-item thinning")
    tail_outputs = generate_tail_item_cut_levels(
        df=train_base,
        item_col=item_col,
        keep_fracs=TAIL_KEEP_FRACS,
        seed=SEED,
        user_col=user_col,
        timestamp_col=timestamp_col,
        protect_first_interaction=True
    )

    tail_report_rows = [{"dataset": "base_train", "keep_frac": 1.0, **train_base_metrics}]

    for level_name, thin_train_df in tail_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_train_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )
        thin_output = output_root / f"{dataset_name}_tail_{level_name}"
        keep_frac = len(thin_train_df) / len(train_base)
        thin_metadata = {
            "dataset_name": dataset_name,
            "method": "tail_item_cut_train_only",
            "parent_dataset": str(base_output),
            "preprocessing_method": preprocessing_method,
            "user_col": user_col,
            "item_col": item_col,
            "timestamp_col": timestamp_col,
            "k_user": k_user,
            "k_item": k_item,
            "test_size": TEST_SIZE,
            "split_method": "userwise_temporal_split",
            "dedup_user_item": DEDUP_USER_ITEM,
            "dedup_keep": "last" if DEDUP_USER_ITEM else None,
            "keep_frac": keep_frac,
            "reference_stats": reference_stats,
            "train_metrics": thin_metrics,
            "test_policy": "fixed_test_from_base_split",
            "test_rows": int(len(test_base)),
        }
        save_train_test_folder(
            train_df=thin_train_df, test_df=test_base,
            output_dir=thin_output, user_col=user_col, item_col=item_col,
            metadata=thin_metadata,
        )
        tail_report_rows.append({"dataset": level_name, "keep_frac": keep_frac, **thin_metrics})

    Path("data/reports").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(random_report_rows).to_csv(
        f"data/reports/{dataset_name}_random_summary.csv", index=False
    )
    pd.DataFrame(head_report_rows).to_csv(
        f"data/reports/{dataset_name}_head_summary.csv", index=False
    )
    pd.DataFrame(tail_report_rows).to_csv(
        f"data/reports/{dataset_name}_tail_summary.csv", index=False
    )

    print("Done.")


if __name__ == "__main__":
    main()
