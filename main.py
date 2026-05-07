from pathlib import Path
import pandas as pd

from src.config import get_dataset_config
from src.io_utils import load_dataframe
from src.kcore import make_k_core
from src.preprocessing import deduplicate_user_item
from src.splitting import userwise_temporal_split
from src.thinning import generate_random_thinning_levels, generate_head_item_cut_levels
from src.metrics import build_reference_stats, compute_sparsity_metrics
from src.dataset_folder import save_dataset_folder, save_train_test_folder

# thay tham số đầu vào
# DATASET_NAME = "baby_product" #hm
# K = 5 
# TEST_SIZE = 0.2
# KEEP_FRACS = [0.9, 0.7, 0.5, 0.3, 0.1]
# SEED = 42
# CUT_RATIO_PER_ROUND = 0.5
# TOP_N_PER_ROUND = 1000 #500
# DEDUP_USER_ITEM = True #chạy version bỏ duplicate interaction (true) hay giữ duplicate (false)

DATASET_NAME = "hm" #hm
K = 10 
TEST_SIZE = 0.2
KEEP_FRACS = [0.9, 0.7, 0.5, 0.3, 0.1]
SEED = 42
CUT_RATIO_PER_ROUND = 0.5
TOP_N_PER_ROUND = 500
DEDUP_USER_ITEM = True #chạy version bỏ duplicate interaction (true) hay giữ duplicate (false)



def main():
    # load dataset
    cfg = get_dataset_config(DATASET_NAME)
    input_path = cfg["input_path"]
    user_col = cfg["user_col"]
    item_col = cfg["item_col"]
    timestamp_col = cfg["timestamp_col"]

    output_root = Path("data/processed")
    output_root.mkdir(parents=True, exist_ok=True)

    print("Step 1: Load raw data")
    df = load_dataframe(input_path)
    print(f"Raw shape: {df.shape}")

    # xử lý interaction trùng theo cấu hình đầu vào
    if DEDUP_USER_ITEM:
        print("Step 2: Deduplicate user-item pairs")
        processed_df = deduplicate_user_item(
            df=df,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            keep='last'
        )
        preprocessing_method = "dedup_user_item"
        version_name = "dedup"

        print(f"After dedup shape: {processed_df.shape}")
        print(f"Removed duplicate rows: {len(df) - len(processed_df)}")

    else:
        print("Step 2: Keep duplicate user-item interactions")
        processed_df = df.copy()
        preprocessing_method = "keep_duplicate_interactions"
        version_name = "transaction"

        print(f"After preprocessing shape: {processed_df.shape}")

    # tạo dense dataset bằng cách áp dụng k-core
    print(f"Step 3: Apply {K}-core")
    dense_df = make_k_core(
        df=processed_df, user_col=user_col, item_col=item_col, k=K, verbose=True
    )

    # tính toán một số thống kê tham chiếu từ dataset dense để hỗ trợ tính toán các metrics về sparsity ở bước tiếp
    reference_stats = build_reference_stats(
        dense_df, user_col=user_col, item_col=item_col
    )

    # tính toán độ thưa của dataset dense sau khi lọc k-core
    dense_metrics = compute_sparsity_metrics(
        dense_df, user_col=user_col, item_col=item_col, reference_stats=reference_stats
    )

    dense_output = output_root / f"{DATASET_NAME}_k{K}_{version_name}"
    dense_metadata = {
        "dataset_name": DATASET_NAME,
        "method": f"{preprocessing_method}_kcore",
        "user_col": user_col,
        "item_col": item_col,
        "timestamp_col": timestamp_col,
        "k": K,
        "dedup_user_item": DEDUP_USER_ITEM,
        "dedup_keep": "last" if DEDUP_USER_ITEM else None,
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

    # chia train/test theo từng user dựa trên timestamp
    print("Step 4: User-wise temporal train/test split")
    train_base, test_base = userwise_temporal_split(
        df=dense_df,
        user_col=user_col,
        timestamp_col=timestamp_col,
        test_size=TEST_SIZE,
        min_train_interactions=1,
        min_test_interactions=1
    )

    print(f"Train base shape: {train_base.shape}")
    print(f"Test base shape: {test_base.shape}")

    # tính toán độ thưa của tập train gốc sau khi split
    train_base_metrics = compute_sparsity_metrics(
        train_base,
        user_col=user_col,
        item_col=item_col,
        reference_stats=reference_stats
    )

    base_output = output_root / f"{DATASET_NAME}_k{K}_{version_name}_base_split"
    base_metadata = {
        "dataset_name": DATASET_NAME,
        "method": "userwise_temporal_split",
        "parent_dataset": str(dense_output),
        "preprocessing_method": preprocessing_method,
        "user_col": user_col,
        "item_col": item_col,
        "timestamp_col": timestamp_col,
        "k": K,
        "test_size": TEST_SIZE,
        "split_method": "userwise_temporal_split",
        "dedup_user_item": DEDUP_USER_ITEM,
        "dedup_keep": "last" if DEDUP_USER_ITEM else None,
        "reference_stats": reference_stats,
        "train_metrics": train_base_metrics,
        "test_rows": int(len(test_base)),
        "test_users": int(test_base[user_col].nunique()),
        "test_items": int(test_base[item_col].nunique()),
    }

    save_train_test_folder(
        train_df=train_base,
        test_df=test_base,
        output_dir=base_output,
        user_col=user_col,
        item_col=item_col,
        metadata=base_metadata,
    )

    # thưa hóa dataset bằng cách random thưa hóa với các mức độ thưa khác nhau dựa trên tập train đã tạo ở bước trước
    print("Step 5: Random thinning")
    thinning_outputs = generate_random_thinning_levels(
        df=train_base, keep_fracs=KEEP_FRACS, seed=SEED
    )

    random_report_rows = []

    random_report_rows.append(
        {"dataset": "base_train", "keep_frac": 1.0, **train_base_metrics}
    )

    # tính toán các metrics về sparsity cho từng tập train thưa được tạo ra
    for level_name, thin_train_df in thinning_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_train_df,
            user_col=user_col,
            item_col=item_col,
            reference_stats=reference_stats,
        )

        thin_output = output_root / f"{DATASET_NAME}_k{K}_{version_name}_random_{level_name}"
        keep_frac = len(thin_train_df) / len(train_base)

        thin_metadata = {
            "dataset_name": DATASET_NAME,
            "method": "random_thinning_train_only",
            "parent_dataset": str(base_output),
            "preprocessing_method": preprocessing_method,
            "user_col": user_col,
            "item_col": item_col,
            "timestamp_col": timestamp_col,
            "k": K,
            "test_size": TEST_SIZE,
            "split_method": "userwise_temporal_split",
            "dedup_user_item": DEDUP_USER_ITEM,
            "dedup_keep": "last" if DEDUP_USER_ITEM else None,
            "keep_frac": keep_frac,
            "reference_stats": reference_stats,
            "train_metrics": thin_metrics,
            "test_policy": "fixed_test_from_base_split",
            "test_rows": int(len(test_base)),
        }

        save_train_test_folder(
            train_df=thin_train_df,
            test_df=test_base,
            output_dir=thin_output,
            user_col=user_col,
            item_col=item_col,
            metadata=thin_metadata,
        )

        random_report_rows.append(
            {
                "dataset": level_name,
                "keep_frac": keep_frac,
                **thin_metrics,
            }
        )

    print("Step 6: Head-item thinning")
    head_outputs = generate_head_item_cut_levels(
        df=train_base,
        item_col=item_col,
        keep_fracs=KEEP_FRACS,
        cut_ratio_per_round=CUT_RATIO_PER_ROUND,
        top_n_per_round=TOP_N_PER_ROUND,
        seed=SEED
    )

    head_report_rows = []

    head_report_rows.append(
        {"dataset": "base_train", "keep_frac": 1.0, **train_base_metrics}
    )

    # tính toán các metrics về sparsity cho từng tập train thưa được tạo ra
    for level_name, thin_train_df in head_outputs.items():
        thin_metrics = compute_sparsity_metrics(
            thin_train_df,
            user_col=user_col,
            item_col=item_col,
            reference_stats=reference_stats
        )

        thin_output = output_root / f"{DATASET_NAME}_k{K}_{version_name}_{level_name}"
        keep_frac = len(thin_train_df) / len(train_base)

        thin_metadata = {
            "dataset_name": DATASET_NAME,
            "method": "head_item_cut_train_only",
            "parent_dataset": str(base_output),
            "preprocessing_method": preprocessing_method,
            "user_col": user_col,
            "item_col": item_col,
            "timestamp_col": timestamp_col,
            "k": K,
            "test_size": TEST_SIZE,
            "split_method": "userwise_temporal_split",
            "dedup_user_item": DEDUP_USER_ITEM,
            "dedup_keep": "last" if DEDUP_USER_ITEM else None,
            "keep_frac": keep_frac,
            "reference_stats": reference_stats,
            "train_metrics": thin_metrics,
            "test_policy": "fixed_test_from_base_split",
            "test_rows": int(len(test_base)),
        }

        save_train_test_folder(
            train_df=thin_train_df,
            test_df=test_base,
            output_dir=thin_output,
            user_col=user_col,
            item_col=item_col,
            metadata=thin_metadata,
        )

        head_report_rows.append(
            {"dataset": level_name, "keep_frac": keep_frac, **thin_metrics}
        )

    Path("data/reports").mkdir(parents=True, exist_ok=True)

    random_report_df = pd.DataFrame(random_report_rows)
    random_report_df.to_csv(
        f"data/reports/{DATASET_NAME}_k{K}_{version_name}_random_train_sparsity_summary.csv",
        index=False
    )

    head_report_df = pd.DataFrame(head_report_rows)
    head_report_df.to_csv(
        f"data/reports/{DATASET_NAME}_k{K}_{version_name}_head_item_train_sparsity_summary.csv",
        index=False
    )

    print("Done.")


if __name__ == "__main__":
    main()