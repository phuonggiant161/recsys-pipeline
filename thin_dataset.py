from __future__ import annotations

import argparse

from src.dataset_folder import load_dataset_folder, save_dataset_folder
from src.thinning import random_thin_interactions
from src.metrics import compute_sparsity_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Randomly thin a dataset folder.")
    parser.add_argument("--input", required=True, help="Input dataset folder")
    parser.add_argument("--keep-frac", type=float, required=True, help="Fraction of interactions to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", required=True, help="Output folder")
    return parser.parse_args()


def main():
    args = parse_args()

    df, metadata = load_dataset_folder(args.input)
    user_col = metadata["user_col"]
    item_col = metadata["item_col"]
    reference_stats = metadata.get("reference_stats")

    thin_df = random_thin_interactions(
        df=df,
        keep_frac=args.keep_frac,
        seed=args.seed
    )

    metrics = compute_sparsity_metrics(
        thin_df,
        user_col=user_col,
        item_col=item_col,
        reference_stats=reference_stats
    )

    output_metadata = {
        "dataset_name": metadata["dataset_name"],
        "method": "random_thinning",
        "parent_folder": args.input,
        "user_col": user_col,
        "item_col": item_col,
        "keep_frac": args.keep_frac,
        "seed": args.seed,
        "reference_stats": reference_stats,
        "metrics": metrics,
    }

    save_dataset_folder(
        df=thin_df,
        output_dir=args.output,
        user_col=user_col,
        item_col=item_col,
        metadata=output_metadata
    )

    print(f"Saved sparse dataset to: {args.output}")


if __name__ == "__main__":
    main()