from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.io_utils import save_json, load_json


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