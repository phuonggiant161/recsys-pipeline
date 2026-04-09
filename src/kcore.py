import pandas as pd

def make_k_core(
    df: pd.DataFrame,
    user_col: str = "customer_id",
    item_col: str = "article_id",
    k: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
   
    core_df = df.copy()
    n_iter = 0

    while True:
        n_iter += 1
        prev_rows = len(core_df)
        prev_users = core_df[user_col].nunique()
        prev_items = core_df[item_col].nunique()

        # Count interactions per user
        user_counts = core_df[user_col].value_counts()
        valid_users = user_counts[user_counts >= k].index
        core_df = core_df[core_df[user_col].isin(valid_users)]

        # Count interactions per item
        item_counts = core_df[item_col].value_counts()
        valid_items = item_counts[item_counts >= k].index
        core_df = core_df[core_df[item_col].isin(valid_items)]

        curr_rows = len(core_df)
        curr_users = core_df[user_col].nunique()
        curr_items = core_df[item_col].nunique()

        if verbose:
            print(
                f"Iter {n_iter}: "
                f"rows={curr_rows:,}, users={curr_users:,}, items={curr_items:,}"
            )

        # Stop when no more rows are removed
        if curr_rows == prev_rows and curr_users == prev_users and curr_items == prev_items:
            break

    return core_df.reset_index(drop=True)
