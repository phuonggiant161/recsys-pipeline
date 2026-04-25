from pathlib import Path
import pandas as pd

from src.config import get_dataset_config
from src.io_utils import load_dataframe
from src.kcore import make_k_core
from src.thinning import generate_random_thinning_levels, generate_tail_item_cut_levels
from src.metrics import build_reference_stats, compute_sparsity_metrics
from src.dataset_folder import save_dataset_folder

# thay tham số đầu vào
DATASET_NAME = "baby_product"
K = 5
KEEP_FRACS = [0.9, 0.7, 0.5, 0.3, 0.1]
SEED = 42


def main():
    # load dataset
    cfg = get_dataset_config(DATASET_NAME)
    input_path = cfg["input_path"]
    user_col = cfg["user_col"]
    item_col = cfg["item_col"]

    output_root = Path("data/processed")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Step 1: Load raw data")
    df = load_dataframe(input_path)
    print(f"Raw shape: {df.shape}")
    # tạo dense dataset bằng cách áp dụng k-core
    print(f"Step 2: Apply {K}-core")
    dense_df = make_k_core(
        df=df, user_col=user_col, item_col=item_col, k=K, verbose=True
    )
    # tính toán một số thống kê tham chiếu từ dataset dense để hỗ trợ tính toán các metrics về sparsity ở bước tiếp
    reference_stats = build_reference_stats(
        dense_df, user_col=user_col, item_col=item_col
    )
    # tính toán độ thưa của dataset gốc
    dense_metrics = compute_sparsity_metrics(
        dense_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
    )

    dense_output = output_root / f"{DATASET_NAME}_k{K}"
    dense_metadata = {
        "dataset_name": DATASET_NAME,
        "method": "kcore",
        "user_col": user_col,
        "item_col": item_col,
        "k": K,
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

    # thưa hóa dataset bằng cách random thưa hóa với các mức độ thưa khác nhau dựa trên dataset dense đã tạo ở bước trước
    print("Step 3: Random thinning")
    thinning_outputs = generate_random_thinning_levels(
        df=dense_df, keep_fracs=KEEP_FRACS, seed=SEED
    )

    random_report_rows = []

    random_report_rows.append(
        {"dataset": "dense_kcore", "keep_frac": 1.0, **dense_metrics}
    )

    #  tính toán các metrics về sparsity cho từng dataset thưa được tạo ra
    for level_name, thin_df in thinning_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_df,
            user_col=user_col,
            item_col=item_col,
            reference_stats=reference_stats,
        )

        thin_output = output_root / f"{DATASET_NAME}_k{K}_{level_name}"
        thin_metadata = {
            "dataset_name": DATASET_NAME,
            "method": "random_thinning",
            "parent_dataset": str(dense_output),
            "user_col": user_col,
            "item_col": item_col,
            "keep_frac": len(thin_df) / len(dense_df),
            "reference_stats": reference_stats,
            "metrics": thin_metrics,
        }

        save_dataset_folder(
            df=thin_df,
            output_dir=thin_output,
            user_col=user_col,
            item_col=item_col,
            metadata=thin_metadata,
        )

        random_report_rows.append(
            {
                "dataset": level_name,
                "keep_frac": len(thin_df) / len(dense_df),
                **thin_metrics,
            }
        )

    print("Step 4: Tail-item thinning")
    tail_outputs = generate_tail_item_cut_levels(
        df=dense_df, item_col=item_col, keep_fracs=KEEP_FRACS
    )

    tail_report_rows = []

    tail_report_rows.append(
        {"dataset": "dense_kcore", "keep_frac": 1.0, **dense_metrics}
    )

    for level_name, thin_df in tail_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
        )

        thin_output = output_root / f"{DATASET_NAME}_k{K}_{level_name}"
        keep_frac = len(thin_df) / len(dense_df)

        thin_metadata = {
            "dataset_name": DATASET_NAME,
            "method": "tail_item_cut",
            "parent_dataset": str(dense_output),
            "user_col": user_col,
            "item_col": item_col,
            "keep_frac": keep_frac,
            "reference_stats": reference_stats,
            "metrics": thin_metrics,
        }

        save_dataset_folder(
            df=thin_df,
            output_dir=thin_output,
            user_col=user_col,
            item_col=item_col,
            metadata=thin_metadata,
        )

        tail_report_rows.append(
            {"dataset": level_name, "keep_frac": keep_frac, **thin_metrics}
        )

    Path("data/reports").mkdir(parents=True, exist_ok=True)

    random_report_df = pd.DataFrame(random_report_rows)
    random_report_df.to_csv(
        f"data/reports/{DATASET_NAME}_k{K}_random_sparsity_summary.csv",
        index=False
    )

    tail_report_df = pd.DataFrame(tail_report_rows)
    tail_report_df.to_csv(
        f"data/reports/{DATASET_NAME}_k{K}_tail_item_sparsity_summary.csv",
        index=False
    )

    print("Done.")

if __name__ == "__main__":
    main()
