from __future__ import annotations

DATASET_CONFIG = {
    "hm": {
        "input_path": "data/raw/hm.csv",
        "user_col": "customer_id",
        "item_col": "article_id",
        "default_k": 100,
    },
    "baby_product": {
        "input_path": "data/raw/baby_product.parquet",
        "user_col": "user_id",
        "item_col": "parent_asin",
        "default_k": 5,
    }
}


def get_dataset_config(dataset_name: str) -> dict:
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_CONFIG[dataset_name]