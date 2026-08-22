from __future__ import annotations

import argparse

from src.config import get_dataset_config
from src.io_utils import load_dataframe
from src.kcore import make_k_core
from src.metrics import build_reference_stats, compute_sparsity_metrics
from src.dataset_folder import save_dataset_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Create k-core dense dataset.")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name")
    parser.add_argument("-k", "--k", type=int, default=None,
                        help="Set both --k-user and --k-item (backward compat)")
    parser.add_argument("--k-user", type=int, default=None, dest="k_user",
                        help="Min interactions per user")
    parser.add_argument("--k-item", type=int, default=None, dest="k_item",
                        help="Min interactions per item")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.k is None and args.k_user is None and args.k_item is None:
        raise ValueError("Provide --k, or --k-user / --k-item.")

    if args.k is not None:
        k_user = args.k
        k_item = args.k
    else:
        if args.k_user is None or args.k_item is None:
            raise ValueError("Provide both --k-user and --k-item (or use --k to set both).")
        k_user = args.k_user
        k_item = args.k_item

    cfg = get_dataset_config(args.dataset)
    input_path = cfg["input_path"]
    user_col   = cfg["user_col"]
    item_col   = cfg["item_col"]

    print("Step 1: Load raw data")
    df = load_dataframe(input_path)
    print(f"  Raw shape: {df.shape}")

    print(f"Step 2: Apply k-core  (k_user={k_user}, k_item={k_item})")
    dense_df = make_k_core(
        df=df,
        user_col=user_col,
        item_col=item_col,
        k_user=k_user,
        k_item=k_item,
        verbose=args.verbose,
    )

    reference_stats = build_reference_stats(dense_df, user_col=user_col, item_col=item_col)
    metrics = compute_sparsity_metrics(
        dense_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
    )

    metadata = {
        "dataset_name": args.dataset,
        "method": "kcore",
        "source_path": input_path,
        "user_col": user_col,
        "item_col": item_col,
        "k_user": k_user,
        "k_item": k_item,
        "reference_stats": reference_stats,
        "metrics": metrics,
    }

    save_dataset_folder(
        df=dense_df,
        output_dir=args.output,
        user_col=user_col,
        item_col=item_col,
        metadata=metadata,
    )

    print(f"Saved dense dataset to: {args.output}")


if __name__ == "__main__":
    main()
