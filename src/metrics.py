import numpy as np
import pandas as pd


def gini(array):
    """
    Compute Gini coefficient for a 1D array of non-negative values.
    """
    x = np.array(array, dtype=float)

    if len(x) == 0:
        return np.nan
    if np.any(x < 0):
        raise ValueError("Gini is not defined for negative values.")
    if np.all(x == 0):
        return 0.0

    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def compute_sparsity_metrics(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    reference_stats: dict | None = None
) -> dict:
    """
    Tính độ thưa theo 3 metrics chính: OSS, ISS và USS

    Notes
    -----
    - OSS = 1 - (Interaction/U x I
        + Interaction: là số cặp user,item (chứ không phải toàn bộ transaction)
        + U * I: số user x số item của data gốc (reference stats)
        

    - USS and ISS:
          USS(u) = 1 - n_u / max_u(n_u)
          ISS(i) = 1 - n_i / max_i(n_i)
      where n_u and n_i are computed on unique (user, item) pairs.
    """

    pair_df = (
        df[[user_col, item_col]]
        .dropna(subset=[user_col, item_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    n_users = int(pair_df[user_col].nunique())
    n_items = int(pair_df[item_col].nunique())
    n_interactions = int(len(pair_df))  # unique (user, item)
# trong trường hợp tính OS, ISS, USS trên data gốc thì ref stat = None, còn tính cho sample sau khi đã thưa hóa thì truyền biến ref stats vào
    if reference_stats is not None:
        denom_users = int(reference_stats["base_n_users"])
        denom_items = int(reference_stats["base_n_items"])
    else:
        denom_users = n_users
        denom_items = n_items

#OSS
    denom = denom_users * denom_items
    density = n_interactions / denom if denom > 0 else np.nan
    overall_sparsity = 1 - density if denom > 0 else np.nan

    user_counts = pair_df.groupby(user_col).size()
    item_counts = pair_df.groupby(item_col).size()

  # uss
    user_uss = 1 - (user_counts / user_counts.max())
  #iss
    item_iss = 1 - (item_counts / item_counts.max())
   

    metrics = {
        # basic counts
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,  # unique user-item pairs

        # denominator info for OSS
        "oss_denom_users": int(denom_users),
        "oss_denom_items": int(denom_items),

        # overall sparsity
        "density": float(density) if pd.notna(density) else np.nan,
        "overall_sparsity": float(overall_sparsity) if pd.notna(overall_sparsity) else np.nan,

        # raw user interaction distribution
        "user_interactions_mean": float(user_counts.mean()) if len(user_counts) else np.nan,
        "user_interactions_median": float(user_counts.median()) if len(user_counts) else np.nan,
        "user_interactions_min": int(user_counts.min()) if len(user_counts) else np.nan,
        "user_interactions_max": int(user_counts.max()) if len(user_counts) else np.nan,
        "user_interactions_gini": float(gini(user_counts.values)) if len(user_counts) else np.nan,

        # raw item interaction distribution
        "item_interactions_mean": float(item_counts.mean()) if len(item_counts) else np.nan,
        "item_interactions_median": float(item_counts.median()) if len(item_counts) else np.nan,
        "item_interactions_min": int(item_counts.min()) if len(item_counts) else np.nan,
        "item_interactions_max": int(item_counts.max()) if len(item_counts) else np.nan,
        "item_interactions_gini": float(gini(item_counts.values)) if len(item_counts) else np.nan,

        # USS summary
        "uss_mean": float(user_uss.mean()) if len(user_uss) else np.nan,
        "uss_median": float(user_uss.median()) if len(user_uss) else np.nan,
        "uss_min": float(user_uss.min()) if len(user_uss) else np.nan,
        "uss_max": float(user_uss.max()) if len(user_uss) else np.nan,
        "uss_gini": float(gini(user_uss.values)) if len(user_uss) else np.nan,

        # ISS summary
        "iss_mean": float(item_iss.mean()) if len(item_iss) else np.nan,
        "iss_median": float(item_iss.median()) if len(item_iss) else np.nan,
        "iss_min": float(item_iss.min()) if len(item_iss) else np.nan,
        "iss_max": float(item_iss.max()) if len(item_iss) else np.nan,
        "iss_gini": float(gini(item_iss.values)) if len(item_iss) else np.nan,
    }

    return metrics


def build_reference_stats(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id"
) -> dict:

    required_cols = [user_col, item_col]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    pair_df = (
        df[[user_col, item_col]]
        .dropna(subset=[user_col, item_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return {
        "base_n_users": int(pair_df[user_col].nunique()),
        "base_n_items": int(pair_df[item_col].nunique()),
    }

