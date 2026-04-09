from pathlib import Path
import pandas as pd

from src.kcore import make_k_core
from src.metrics import build_reference_stats, compute_sparsity_metrics


DATASET_NAME = "hm" # đổi thành "baby_product" khi muốn chạy bộ kia

DATASET_CONFIG = {
    "hm": {
        "input_path": "data/raw/hm.csv",
        "user_col": "customer_id",
        "item_col": "article_id",
        "k": 100
    },
    "baby_product": {
        "input_path": "data/raw/baby_product.parquet",
        "user_col": "user_id",
        "item_col": "parent_asin",
        "k": 5
    }
}

cfg = DATASET_CONFIG[DATASET_NAME]

INPUT_PATH = cfg["input_path"]
USER_COL = cfg["user_col"]
ITEM_COL = cfg["item_col"]
K = cfg["k"]
VERBOSE = True

OUTPUT_DATA_PATH = f"data/interim/{DATASET_NAME}_dense_output.csv"
OUTPUT_REPORT_PATH = f"data/reports/{DATASET_NAME}_dense_report.csv"


def load_data(path: str) -> pd.DataFrame:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Không tìm thấy file input: {path_obj}")

    suffix = path_obj.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path_obj)
    elif suffix == ".parquet":
        return pd.read_parquet(path_obj)
    else:
        raise ValueError(f"Định dạng file chưa hỗ trợ: {suffix}")


def save_report(report: dict, path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([report]).to_csv(output_path, index=False)


def main():
    raw_df = load_data(INPUT_PATH)

    required_cols = [USER_COL, ITEM_COL]
    missing_cols = [col for col in required_cols if col not in raw_df.columns]
    if missing_cols:
        raise ValueError(f"Thiếu cột bắt buộc: {missing_cols}")

    print(f"Running dataset: {DATASET_NAME}")
    print("Step 1: Dày hóa dữ liệu bằng k-core để tạo base dataset")

#dataset sau khi đã làm dày hóa --> đây mới là dataset base của toàn bộ experiment 
    dense_df = make_k_core(
        raw_df,
        user_col=USER_COL,
        item_col=ITEM_COL,
        k=K,
        verbose=VERBOSE
    )

    print("Step 2: Build reference stats từ dense dataset")
    reference_stats = build_reference_stats(
        dense_df,
        user_col=USER_COL,
        item_col=ITEM_COL
    )

    print("Step 3: Tính metrics của dense dataset (base)")
    dense_stats = compute_sparsity_metrics(
        dense_df,
        reference_stats=reference_stats,
        user_col=USER_COL,
        item_col=ITEM_COL
    )

    print("Dense dataset stats:")
    for key, value in dense_stats.items():
        print(f"  {key}: {value}")

    Path("data/interim").mkdir(parents=True, exist_ok=True)
    Path("data/reports").mkdir(parents=True, exist_ok=True)

    dense_df.to_csv(OUTPUT_DATA_PATH, index=False)

    report = {
        "dataset_name": DATASET_NAME,
        "input_path": INPUT_PATH,
        "output_data_path": OUTPUT_DATA_PATH,
        "user_col": USER_COL,
        "item_col": ITEM_COL,
        "k": K,
    }

    for key, value in dense_stats.items():
        report[f"dense_{key}"] = value

    save_report(report, OUTPUT_REPORT_PATH)

    print(f"Output Data: {OUTPUT_DATA_PATH}")
    print(f"Report: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    main()