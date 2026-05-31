import json
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"


def load_columns(dataset_dir: Path):
    metadata_path = dataset_dir / "metadata.json"

    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        user_col = metadata.get("user_col")
        item_col = metadata.get("item_col")

        if user_col and item_col:
            return user_col, item_col

    # fallback cho H&M nếu metadata không có thông tin cột
    return "customer_id", "article_id"


def export_elliot_tsv(dataset_dir: Path):
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print(f"[SKIP] {dataset_dir.name} | thiếu train.csv hoặc test.csv")
        return

    user_col, item_col = load_columns(dataset_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_elliot = train_df[[user_col, item_col]].drop_duplicates().copy()
    test_elliot = test_df[[user_col, item_col]].drop_duplicates().copy()

    train_elliot["rating"] = 1
    test_elliot["rating"] = 1

    train_elliot.to_csv(
        dataset_dir / "train.tsv",
        sep="\t",
        index=False,
        header=False
    )

    test_elliot.to_csv(
        dataset_dir / "test.tsv",
        sep="\t",
        index=False,
        header=False
    )

    print(f"[DONE] {dataset_dir.name}")
    print(f"       train.tsv: {len(train_elliot):,} rows")
    print(f"       test.tsv : {len(test_elliot):,} rows")


def main():
    for dataset_dir in PROCESSED_ROOT.iterdir():
        if dataset_dir.is_dir():
            export_elliot_tsv(dataset_dir)


if __name__ == "__main__":
    main()