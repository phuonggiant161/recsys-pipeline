import numpy as np
import pandas as pd

#hàm tính gini để đo distribution
def gini(array):
    x = np.array(array, dtype=float)

    if np.any(x < 0):
        raise ValueError("Gini is not defined for negative values.")

    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

#lấy interaction, user, item counts của data gốc (data trước khi bị làm thưa hóa) để làm reference khi tính độ thưa
def build_reference_stats(
    df: pd.DataFrame,
    user_col: str = "customer_id",
    item_col: str = "article_id"
) -> dict:
    user_counts = df.groupby(user_col).size()
    item_counts = df.groupby(item_col).size()

    return {
        "base_n_users": df[user_col].nunique(),
        "base_n_items": df[item_col].nunique(),
        "base_n_interactions": len(df),
        "base_user_ids": user_counts.index,
        "base_item_ids": item_counts.index,
        "base_max_user_inter": user_counts.max(),
        "base_max_item_inter": item_counts.max(),
    }


def compute_sparsity_metrics(
    df: pd.DataFrame,
    reference_stats: dict,
    user_col: str = "customer_id",
    item_col: str = "article_id"
) -> dict:

    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_interactions = len(df)

    user_counts = df.groupby(user_col).size()
    item_counts = df.groupby(item_col).size()

#chỗ này align user với dataset gốc, tránh việc tính trực tiếp trên current dataframe thì sẽ bỏ qua những user/item đã bị loại bỏ do thưa hóa, dẫn đến việc đánh giá sai độ thưa so với dataset gốc
    aligned_user_counts = user_counts.reindex(reference_stats["base_user_ids"], fill_value=0)
    aligned_item_counts = item_counts.reindex(reference_stats["base_item_ids"], fill_value=0)

#OS is the only formula that can be calculated overall, while rest are calculated per user/item --> using median, avg to summarize
    os_val = 1 - (n_interactions / reference_stats["base_n_interactions"])
    uss = 1 - (aligned_user_counts / reference_stats["base_max_user_inter"])
    iss = 1 - (aligned_item_counts / reference_stats["base_max_item_inter"])

    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,

        "OS": os_val,
        "mean_uss": uss.mean(),
        "median_uss": uss.median(),
        "mean_iss": iss.mean(),
        "median_iss": iss.median(),

        "avg_inter_per_user": user_counts.mean(),
        "median_inter_per_user": user_counts.median(),
        "min_inter_per_user": user_counts.min(),
        "max_inter_per_user": user_counts.max(),

        "avg_inter_per_item": item_counts.mean(),
        "median_inter_per_item": item_counts.median(),
        "min_inter_per_item": item_counts.min(),
        "max_inter_per_item": item_counts.max(),

        "gini_user": gini(user_counts.values),
        "gini_item": gini(item_counts.values),
    }


