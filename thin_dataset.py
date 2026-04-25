import argparse
from pathlib import Path

import pandas as pd

from src.dataset_folder import load_dataset_folder, save_dataset_folder
from src.metrics import compute_sparsity_metrics
from src.thinning import random_thin_interactions, tail_item_cut


def parse_args():
    parser = argparse.ArgumentParser(description="Thin a prepared dataset folder.")
    parser.add_argument("--input", required=True, help="Input dataset folder")
    parser.add_argument("--output", required=True, help="Output dataset folder")
    parser.add_argument(
        "--method",
        choices=["random", "tail_item"],
        default="random",
        help="Thinning method",
    )
    parser.add_argument(
        "--keep-frac",
        type=float,
        required=True,
        help="Fraction of interactions/records to keep",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed, only used for random thinning")
    return parser.parse_args()


def validate_keep_frac(keep_frac: float) -> None:
    if not 0 < keep_frac <= 1:
        raise ValueError("keep_frac must be in the range (0, 1].")


def thin_dataset(
    df: pd.DataFrame,
    method: str,
    keep_frac: float,
    item_col: str,
    seed: int,
) -> pd.DataFrame:
    validate_keep_frac(keep_frac)

    if method == "random":
        return random_thin_interactions(df=df, keep_frac=keep_frac, seed=seed)

    if method == "tail_item":
        return tail_item_cut(df=df, item_col=item_col, keep_frac=keep_frac)

    raise ValueError(f"Unsupported thinning method: {method}")


def main():
    args = parse_args()

    df, metadata = load_dataset_folder(args.input)
    user_col = metadata["user_col"]
    item_col = metadata["item_col"]
    reference_stats = metadata.get("reference_stats")

    thin_df = thin_dataset(
        df=df,
        method=args.method,
        keep_frac=args.keep_frac,
        item_col=item_col,
        seed=args.seed,
    )

    metrics = compute_sparsity_metrics(
        thin_df,
        user_col=user_col,
        item_col=item_col,
        reference_stats=reference_stats,
    )

    actual_keep_frac = len(thin_df) / len(df) if len(df) > 0 else 0
    method_name = "random_thinning" if args.method == "random" else "tail_item_cut"

    output_metadata = {
        "dataset_name": metadata.get("dataset_name"),
        "method": method_name,
        "parent_dataset": str(Path(args.input)),
        "user_col": user_col,
        "item_col": item_col,
        "requested_keep_frac": args.keep_frac,
        "actual_keep_frac": actual_keep_frac,
        "seed": args.seed if args.method == "random" else None,
        "reference_stats": reference_stats,
        "metrics": metrics,
    }

    save_dataset_folder(
        df=thin_df,
        output_dir=args.output,
        user_col=user_col,
        item_col=item_col,
        metadata=output_metadata,
    )

    print(f"Saved {method_name} dataset to: {args.output}")
    print(f"Requested keep_frac: {args.keep_frac}")
    print(f"Actual keep_frac: {actual_keep_frac:.6f}")


if __name__ == "__main__":
    main()
