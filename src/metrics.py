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
    
    #đếm số interaction group by theo user và item
    user_counts = pair_df.groupby(user_col).size()
    item_counts = pair_df.groupby(item_col).size()

    #đếm tổng
    n_users = int(pair_df[user_col].nunique())  #số user tại current dataset
    n_items = int(pair_df[item_col].nunique())  #số item tại current dataset
    n_interactions = int(len(pair_df)) #đây là tử số trong OSS # unique (user, item)
    n_rows = int(len(df)) #số lượng transaction

# trong trường hợp tính OS, ISS, USS trên data gốc thì ref stat = None, còn tính cho sample sau khi đã thưa hóa thì truyền biến ref stats vào
    if reference_stats is not None:
        denom_users = int(reference_stats["base_n_users"]) 
        denom_items = int(reference_stats["base_n_items"])
        denom_rows = int(reference_stats["base_n_rows"])
    else:
        denom_users = n_users
        denom_items = n_items
        denom_rows = int(len(df))

#OSS
 # Cách 1: unique current interactions (1 cặp U,I) / reference U*I
    denom_ui = denom_users * denom_items

    oss_c1 = (
        n_interactions / denom_ui
        if denom_ui > 0
        else np.nan
    )
# OSS method 2:
# unique current interactions / current raw row count
    oss_c2 = (
        n_interactions / denom_rows
        if denom_rows > 0
        else np.nan
    )

  # uss
    user_uss = n_interactions/denom_users
  #iss
    item_iss = n_interactions/denom_items

  # -------------------------
    # User / item cold-start
    # -------------------------
    # coldstart_user = % users from dense baseline missing in current dataset
    # coldstart_item = % items from dense baseline missing in current dataset
   
    coldstart_user = 1 - (n_users / denom_users)
    coldstart_item = 1 - (n_items / denom_items)

    metrics = {
        # current dataset counts
        "n_rows": n_rows,
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_interactions,

        # reference denominators
        #"oss_denom_users": int(denom_users),
        #"oss_denom_items": int(denom_items),
        #"oss_denom_ui": int(denom_ui),
        #"oss_denom_rows": int(denom_rows),

        # OSS C1: current unique interactions / reference U*I
        "oss_c1": float(oss_c1) if pd.notna(oss_c1) else np.nan,
        # "oss_c1_pct": float(oss_c1 * 100) if pd.notna(oss_c1) else np.nan,

        # OSS C2: current unique interactions / reference len(df)
        "oss_c2": float(oss_c2) if pd.notna(oss_c2) else np.nan,
        # "oss_c2_pct": float(oss_c2 * 100) if pd.notna(oss_c2) else np.nan,

        # USS / ISS
        "uss": float(user_uss) if pd.notna(user_uss) else np.nan,
        "iss": float(item_iss) if pd.notna(item_iss) else np.nan,

        #gini
        "user_gini": float(gini(user_counts.values)) if len(user_counts) else np.nan,
        "item_gini": float(gini(item_counts.values)) if len(item_counts) else np.nan,

        # cold-start
        "coldstart_user": float(coldstart_user) if pd.notna(coldstart_user) else np.nan,
        "coldstart_item": float(coldstart_item) if pd.notna(coldstart_item) else np.nan,
        "pct_coldstart_user": (float(coldstart_user * 100) if pd.notna(coldstart_user) else np.nan),
        "pct_coldstart_item": (float(coldstart_item * 100) if pd.notna(coldstart_item) else np.nan),
    }

    return metrics


def build_reference_stats(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id"
) -> dict:
    """Tính số user, item của data gốc để làm mẫu số cho OSS, USS, ISS khi tính độ thưa của sample dataset sau khi đã thưa hóa"""

    pair_df = (
        df[[user_col, item_col]]
        .dropna(subset=[user_col, item_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return {
        "base_n_rows": int(len(df)),  # số dòng raw của reference data
        "base_n_users": int(pair_df[user_col].nunique()),
        "base_n_items": int(pair_df[item_col].nunique()),
        "base_n_interactions": int(len(pair_df)),  # unique (user, item), để check thêm nếu cần
    }


