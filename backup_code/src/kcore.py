import pandas as pd


def make_k_core(
    df: pd.DataFrame,
    user_col: str = "customer_id",
    item_col: str = "article_id",
    k_user: int = 5,
    k_item: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Iteratively remove users with < k_user interactions and items with < k_item interactions."""
    core_df = df.copy()
    n_iter = 0

    rows_before = len(core_df)
    users_before = core_df[user_col].nunique()
    items_before = core_df[item_col].nunique()

    while True:
        n_iter += 1
        prev_len = len(core_df)

        user_counts = core_df[user_col].value_counts()
        valid_users = user_counts[user_counts >= k_user].index
        core_df = core_df[core_df[user_col].isin(valid_users)]

        item_counts = core_df[item_col].value_counts()
        valid_items = item_counts[item_counts >= k_item].index
        core_df = core_df[core_df[item_col].isin(valid_items)]

        if verbose:
            print(
                f"  Iter {n_iter}: "
                f"rows={len(core_df):,}, "
                f"users={core_df[user_col].nunique():,}, "
                f"items={core_df[item_col].nunique():,}"
            )

        if len(core_df) == prev_len:
            break

    rows_after = len(core_df)
    users_after = core_df[user_col].nunique()
    items_after = core_df[item_col].nunique()

    print(
        f"\n  k-core summary  (k_user={k_user}, k_item={k_item}, iters={n_iter})\n"
        f"  {'':10s}  {'rows':>12s}  {'users':>10s}  {'items':>10s}\n"
        f"  {'before':10s}  {rows_before:>12,}  {users_before:>10,}  {items_before:>10,}\n"
        f"  {'after':10s}  {rows_after:>12,}  {users_after:>10,}  {items_after:>10,}\n"
        f"  {'removed':10s}  {rows_before - rows_after:>12,}  "
        f"{users_before - users_after:>10,}  {items_before - items_after:>10,}"
    )

    return core_df.reset_index(drop=True)
