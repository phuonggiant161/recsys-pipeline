from typing import Optional

import numpy as np
import pandas as pd


def gini(array):
    """Compute Gini coefficient for a 1D array of non-negative values."""
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


def build_reference_stats(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> dict:
    """Build reference population stats from a base train DataFrame.

    Call this on the final base train set (train_base after train/valid split).
    All thinning-level metrics are then expressed relative to the un-thinned
    train population, so keep_frac=1.0 is the natural baseline.
    """
    pair_df = (
        df[[user_col, item_col]]
        .dropna(subset=[user_col, item_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return {
        "base_n_users":        int(pair_df[user_col].nunique()),
        "base_n_items":        int(pair_df[item_col].nunique()),
        "base_n_interactions": int(len(pair_df)),
        "base_users":          pair_df[user_col].drop_duplicates().tolist(),
        "base_items":          pair_df[item_col].drop_duplicates().tolist(),
    }


def compute_sparsity_metrics(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    reference_stats: Optional[dict] = None,
) -> dict:
    """Compute sparsity metrics for a train DataFrame.

    All metrics are relative to the reference population (base_users / base_items).
    Pass reference_stats=None to use df's own population as reference.

    Metrics
    -------
    n_interactions : int
        Unique (user, item) pairs in the current train set.

    oss : float
        Observed density = n_interactions / (base_n_users * base_n_items).
        This IS the density fraction. NOT "1 - density".
        Display as percentage: oss * 100.

    uss : float
        Average observed interactions per reference user
        = n_interactions / base_n_users.
        NOT "1 - n/max(n)".

    iss : float
        Average observed interactions per reference item
        = n_interactions / base_n_items.
        NOT "1 - n/max(n)".

    user_gini / item_gini : float
        Gini coefficient of interaction counts over ALL reference users/items,
        zero-filled for those absent from the current train set.

    coldstart_user / coldstart_item : float
        Proportion of reference users/items with <= 1 interaction in current
        train. Includes users/items removed entirely (0 interactions).
    """
    pair_df = (
        df[[user_col, item_col]]
        .dropna(subset=[user_col, item_col])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    user_counts = pair_df.groupby(user_col).size()
    item_counts = pair_df.groupby(item_col).size()

    n_interactions = int(len(pair_df))
    n_users        = int(pair_df[user_col].nunique())
    n_items        = int(pair_df[item_col].nunique())
    n_rows         = int(len(df))

    if reference_stats is not None:
        denom_users = int(reference_stats["base_n_users"])
        denom_items = int(reference_stats["base_n_items"])
        base_users  = reference_stats["base_users"]
        base_items  = reference_stats["base_items"]
    else:
        denom_users = n_users
        denom_items = n_items
        base_users  = pair_df[user_col].drop_duplicates().tolist()
        base_items  = pair_df[item_col].drop_duplicates().tolist()

    # Observed density (not 1-density)
    denom_ui = denom_users * denom_items
    oss = n_interactions / denom_ui if denom_ui > 0 else np.nan

    # Average interactions per reference user / item
    uss = n_interactions / denom_users if denom_users > 0 else np.nan
    iss = n_interactions / denom_items if denom_items > 0 else np.nan

    # Zero-fill counts over the full reference population
    user_counts_full = user_counts.reindex(base_users, fill_value=0)
    item_counts_full = item_counts.reindex(base_items, fill_value=0)

    # Cold-start: proportion of reference users/items with <= 1 interaction
    coldstart_user = float((user_counts_full <= 1).mean())
    coldstart_item = float((item_counts_full <= 1).mean())

    return {
        # raw counts (kept for metadata.json storage)
        "n_rows":         n_rows,
        "n_users":        n_users,
        "n_items":        n_items,
        "n_interactions": n_interactions,
        # sparsity metrics
        "oss":            float(oss)  if pd.notna(oss)  else np.nan,
        "uss":            float(uss)  if pd.notna(uss)  else np.nan,
        "iss":            float(iss)  if pd.notna(iss)  else np.nan,
        "user_gini":      float(gini(user_counts_full.values)),
        "item_gini":      float(gini(item_counts_full.values)),
        "coldstart_user": coldstart_user,
        "coldstart_item": coldstart_item,
    }
