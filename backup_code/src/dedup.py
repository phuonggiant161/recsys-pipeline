import pandas as pd


def deduplicate_user_item(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    keep: str = "last",
) -> pd.DataFrame:
    """
    - loại bỏ các bản ghi trùng lặp của user-item pair, chỉ giữ lại 1 bản ghi duy nhất cho mỗi cặp user-item.
    - giữ lại bản ghi mới nhất (keep="last") dựa trên timestamp.  

    """

    required_cols = [user_col, item_col, timestamp_col]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    dedup_df = (
        df.dropna(subset=[user_col, item_col, timestamp_col])
          .copy()
    )

    dedup_df[timestamp_col] = pd.to_datetime(
        dedup_df[timestamp_col],
        errors="coerce"
    )
#drop nếu timestamp không hợp lệ, sắp xếp theo user và timestamp, giữ lại bản ghi mới nhất của mỗi cặp user-item
    dedup_df = (
        dedup_df.dropna(subset=[timestamp_col])
                .sort_values(
                    by=[user_col, timestamp_col],
                    ascending=[True, True],
                    kind="mergesort"
                )
                .drop_duplicates(
                    subset=[user_col, item_col],
                    keep=keep #keep="last" để giữ lại bản ghi mới nhất
                )
                .reset_index(drop=True)
    )

    return dedup_df