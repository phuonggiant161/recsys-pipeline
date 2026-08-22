import argparse
from pathlib import Path
import pandas as pd

from src.dataset_folder import load_dataset_folder
from src.metrics import compute_sparsity_metrics
from src.io_utils import save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    df, metadata = load_dataset_folder(args.input)
    user_col = metadata["user_col"]
    item_col = metadata["item_col"]
    reference_stats = metadata.get("reference_stats")

    metrics = compute_sparsity_metrics(
        df=df,
        user_col=user_col,
        item_col=item_col,
        reference_stats=reference_stats
    )

    print("Sparsity metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    report_dir = Path(args.input)
    pd.DataFrame([metrics]).to_csv(report_dir / "sparsity_report.csv", index=False)

    print(f"Saved reports to: {report_dir}")


if __name__ == "__main__":
    main()