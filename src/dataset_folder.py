import json
from pathlib import Path

import pandas as pd


def save_json(data: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_json(input_path: str | Path) -> dict:
    input_path = Path(input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset_folder(
    df: pd.DataFrame,
    output_dir: str | Path,
    user_col: str,
    item_col: str,
    metadata: dict
) -> None:

    """Lưu dataset đã chuẩn hóa thành folder với 4 file:
- interactions.csv: chứa tất cả các record tương tác (customer_id, article_id)
- users.csv: chứa danh sách người dùng
- items.csv: chứa danh sách sản phẩm
- metadata.json: chứa thông tin metadata bổ sung (nếu có)"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = output_dir / "interactions.csv"
    users_path = output_dir / "users.csv"
    items_path = output_dir / "items.csv"
    metadata_path = output_dir / "metadata.json"

    df.to_csv(interactions_path, index=False)

    (
        pd.DataFrame({user_col: sorted(df[user_col].dropna().unique())})
        .to_csv(users_path, index=False)
    )

    (
        pd.DataFrame({item_col: sorted(df[item_col].dropna().unique())})
        .to_csv(items_path, index=False)
    )

    save_json(metadata, metadata_path)


def load_dataset_folder(input_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    input_dir = Path(input_dir)

    interactions_path = input_dir / "interactions.csv"
    metadata_path = input_dir / "metadata.json"

    df = pd.read_csv(interactions_path)
    metadata = load_json(metadata_path)

    return df, metadata


def save_train_test_folder(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
    user_col: str,
    item_col: str,
    metadata: dict,
) -> None:
    """Lưu dataset đã chia train/test thành folder với 5 file:
- train.csv: chứa các record dùng để train model
- test.csv: chứa các record dùng để evaluate model
- users.csv: chứa danh sách người dùng trong train và test
- items.csv: chứa danh sách sản phẩm trong train và test
- metadata.json: chứa thông tin metadata bổ sung (nếu có)"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    users_path = output_dir / "users.csv"
    items_path = output_dir / "items.csv"
    metadata_path = output_dir / "metadata.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # chuyển sang format chuẩn tsv của elliot
    train_elliot = train_df[[user_col, item_col]].drop_duplicates().copy()
    test_elliot = test_df[[user_col, item_col]].drop_duplicates().copy()

    train_elliot["rating"] = 1
    test_elliot["rating"] = 1

    train_elliot.to_csv(
        output_dir / "train.tsv",
        sep="\t",
        index=False,
        header=False
    )

    test_elliot.to_csv(
        output_dir / "test.tsv",
        sep="\t",
        index=False,
        header=False
    )

    all_df = pd.concat([train_df, test_df], ignore_index=True)

    (
        pd.DataFrame({user_col: sorted(all_df[user_col].dropna().unique())})
        .to_csv(users_path, index=False)
    )

    (
        pd.DataFrame({item_col: sorted(all_df[item_col].dropna().unique())})
        .to_csv(items_path, index=False)
    )

    save_json(metadata, metadata_path)

