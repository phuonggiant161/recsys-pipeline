"""
Thinning strategies for interaction datasets.

Public API
----------
build_protected_interactions(df, user_col, item_col, timestamp_col)
    Returns the minimal subset of df that must stay in train to cover every
    user and every item at least once.

thin_with_protected_user_item_interactions(
    df, user_col, item_col, timestamp_col, keep_frac, thinning_func, ...)
    Wraps any thinning strategy so the result is guaranteed to contain the
    same user set and item set as the input.

generate_random_thinning_levels(df, keep_fracs, user_col, item_col, timestamp_col, ...)
generate_head_item_cut_levels(df, keep_fracs, user_col, item_col, timestamp_col, ...)
generate_tail_item_cut_levels(df, keep_fracs, user_col, item_col, timestamp_col, ...)
    Batch helpers that return {level_name: thinned_df}.

random_thin_interactions(df, keep_frac, seed)
head_item_cut(df, item_col, keep_frac, seed, ...)
tail_item_cut(df, item_col, keep_frac, seed, ...)
    Pure thinning functions — no user/item protection, no assertions.
    Suitable as thinning_func arguments.
"""

import math
import pandas as pd
import numpy as np
from time import perf_counter


# ---------------------------------------------------------------------------
# Protected interaction helpers
# ---------------------------------------------------------------------------

def build_protected_row_ids(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
) -> frozenset:
    """
    Identify the minimal set of row IDs that must be kept in train so that
    every user and every item has at least one interaction.

    df must contain a column named "__row_id" with unique integer values
    (typically 0-indexed).  The caller is responsible for creating it.

    Algorithm
    ---------
    Step 1 — For each user, select the row with the earliest timestamp.
             Ties are broken by __row_id (stable across runs).

    Step 2 — Determine which items are *not* already covered by step-1 rows.
             For each such item, select the row with the earliest timestamp.

    Step 3 — Return the union of both sets (no duplicates).

    Returns
    -------
    frozenset of __row_id integer values.
    """
    assert "__row_id" in df.columns, "df must have a '__row_id' column"

    # Single sort used for both steps: oldest first, row_id breaks ties
    sorted_df = df.sort_values([timestamp_col, "__row_id"], kind="stable")

    # Step 1: first interaction per user
    user_first_ids: set = set(
        sorted_df.groupby(user_col, sort=False)["__row_id"].first().values
    )

    # Step 2: items not covered by step-1
    covered_items: set = set(
        df.loc[df["__row_id"].isin(user_first_ids), item_col]
    )
    uncovered_df = sorted_df.loc[~sorted_df[item_col].isin(covered_items)]

    item_first_ids: set = set()
    if len(uncovered_df) > 0:
        item_first_ids = set(
            uncovered_df.groupby(item_col, sort=False)["__row_id"].first().values
        )

    return frozenset(user_first_ids | item_first_ids)


def build_protected_interactions(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    Return the subset of df that must stay in train to cover every user and
    every item.  See build_protected_row_ids() for the selection algorithm.

    This is a convenience wrapper that manages the temporary __row_id column.
    """
    work = df.reset_index(drop=True).copy()
    work["__row_id"] = np.arange(len(work))

    protected_ids = build_protected_row_ids(work, user_col, item_col, timestamp_col)

    return (
        work.loc[work["__row_id"].isin(protected_ids)]
            .drop(columns="__row_id")
            .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Thinning with user+item preservation
# ---------------------------------------------------------------------------

def thin_with_protected_user_item_interactions(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    keep_frac: float,
    thinning_func,
    seed: int = 42,
    thinning_kwargs: dict | None = None,
) -> pd.DataFrame:
    """
    Apply thinning_func while guaranteeing the result contains every user
    and every item from the input df.

    Strategy
    --------
    1. Compute the protected set (first per user + first per uncovered item).
    2. Thin only the non-protected remainder, adjusting the fraction so that
       the total kept rows equals int(round(len(df) * keep_frac)).
    3. Concatenate protected + thinned remainder.
    4. Assert user set and item set are fully preserved.

    Parameters
    ----------
    df            : Input interaction DataFrame.
    user_col      : Column name for user IDs.
    item_col      : Column name for item IDs.
    timestamp_col : Column name for timestamps.
    keep_frac     : Target fraction of rows to keep (0 < keep_frac ≤ 1).
    thinning_func : Callable f(df, keep_frac, seed, **kwargs) → DataFrame.
    seed          : Random seed passed through to thinning_func.
    thinning_kwargs: Extra keyword arguments forwarded to thinning_func.

    Returns
    -------
    DataFrame with the same columns as df, same user set, same item set.

    Raises
    ------
    ValueError    if keep_frac is too small to cover all protected rows.
    AssertionError if the user/item sets are somehow not preserved.
    """
    if thinning_kwargs is None:
        thinning_kwargs = {}

    work = df.reset_index(drop=True).copy()
    work["__row_id"] = np.arange(len(work))

    protected_ids = build_protected_row_ids(work, user_col, item_col, timestamp_col)
    n_protected = len(protected_ids)
    target_total = int(round(len(df) * keep_frac))

    if target_total < n_protected:
        raise ValueError(
            f"keep_frac={keep_frac:.4f} is too small to cover all users and items. "
            f"Need ≥ {n_protected} rows "
            f"({df[user_col].nunique()} users + uncovered items), "
            f"but target_total={target_total}. "
            f"Minimum keep_frac ≈ {n_protected / len(df):.4f}."
        )

    protected_df = (
        work.loc[work["__row_id"].isin(protected_ids)]
            .drop(columns="__row_id")
            .reset_index(drop=True)
    )
    remainder_df = (
        work.loc[~work["__row_id"].isin(protected_ids)]
            .drop(columns="__row_id")
            .reset_index(drop=True)
    )

    target_remainder = target_total - n_protected

    if target_remainder == 0 or len(remainder_df) == 0:
        thinned_remainder = remainder_df.iloc[0:0].copy()
    elif target_remainder >= len(remainder_df):
        thinned_remainder = remainder_df.copy()
    else:
        adj_frac = target_remainder / len(remainder_df)
        thinned_remainder = thinning_func(
            df=remainder_df,
            keep_frac=adj_frac,
            seed=seed,
            **thinning_kwargs,
        )

    result = (
        pd.concat([protected_df, thinned_remainder], ignore_index=True)
          .reset_index(drop=True)
    )

    assert set(result[user_col]) == set(df[user_col]), (
        "BUG: user set not preserved after thinning. "
        f"Missing: {set(df[user_col]) - set(result[user_col])}"
    )
    assert set(result[item_col]) == set(df[item_col]), (
        "BUG: item set not preserved after thinning. "
        f"Missing: {set(df[item_col]) - set(result[item_col])}"
    )

    return result


# ---------------------------------------------------------------------------
# Batch thinning helpers
# ---------------------------------------------------------------------------

def generate_random_thinning_levels(
    df: pd.DataFrame,
    keep_fracs: list[float],
    user_col: str,
    item_col: str,
    timestamp_col: str,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Produce multiple random-thinned datasets, one per keep_frac.

    All outputs share the same user set and item set as df.

    Returns
    -------
    {level_name: thinned_df}  e.g. {"keep0.1": df_thin, "keep0.5": df_thin2}
    """
    outputs: dict[str, pd.DataFrame] = {}
    for i, kf in enumerate(keep_fracs, start=1):
        level_name = f"keep{kf:g}"
        outputs[level_name] = thin_with_protected_user_item_interactions(
            df=df,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            keep_frac=kf,
            thinning_func=random_thin_interactions,
            seed=seed + i,
        )
    return outputs


def generate_head_item_cut_levels(
    df: pd.DataFrame,
    keep_fracs: list[float],
    user_col: str,
    item_col: str,
    timestamp_col: str,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Produce multiple head-item-cut datasets, one per keep_frac.

    All outputs share the same user set and item set as df.
    """
    outputs: dict[str, pd.DataFrame] = {}
    for i, kf in enumerate(keep_fracs, start=1):
        level_name = f"keep{kf:g}"
        outputs[level_name] = thin_with_protected_user_item_interactions(
            df=df,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            keep_frac=kf,
            thinning_func=head_item_cut,
            seed=seed + i,
            thinning_kwargs={"item_col": item_col, "verbose": verbose},
        )
    return outputs


def generate_tail_item_cut_levels(
    df: pd.DataFrame,
    keep_fracs: list[float],
    user_col: str,
    item_col: str,
    timestamp_col: str,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Produce multiple tail-item-cut datasets, one per keep_frac.

    All outputs share the same user set and item set as df.
    """
    outputs: dict[str, pd.DataFrame] = {}
    for i, kf in enumerate(keep_fracs, start=1):
        level_name = f"keep{kf:g}"
        outputs[level_name] = thin_with_protected_user_item_interactions(
            df=df,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            keep_frac=kf,
            thinning_func=tail_item_cut,
            seed=seed + i,
            thinning_kwargs={"item_col": item_col, "verbose": verbose},
        )
    return outputs


# ---------------------------------------------------------------------------
# Pure thinning functions (no protection, no assertions)
# ---------------------------------------------------------------------------

def random_thin_interactions(
    df: pd.DataFrame,
    keep_frac: float,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Randomly keep keep_frac of rows.  No user/item protection.
    """
    n_keep = int(round(len(df) * keep_frac))
    return (
        df.sample(n=n_keep, random_state=seed, replace=False)
          .reset_index(drop=True)
    )


def head_item_cut(
    df: pd.DataFrame,
    item_col: str,
    keep_frac: float,
    seed: int = 42,
    verbose: bool = True,
    cap_log_every: int = 1000,
) -> pd.DataFrame:
    """
    Cap the number of interactions per item to reduce head-heavy items.

    Finds the smallest cap such that keeping at most cap interactions per
    item yields >= target rows.  Items at the exact cap boundary are randomly
    sub-sampled to hit the target precisely.

    No user/item protection — use thin_with_protected_user_item_interactions
    as the outer wrapper when protection is needed.
    """
    t_all = perf_counter()
    n_keep = int(round(len(df) * keep_frac))

    if verbose:
        print(
            f"[HEAD] keep_frac={keep_frac:.2f} | rows={len(df):,} | "
            f"target={n_keep:,}",
            flush=True,
        )

    if n_keep >= len(df):
        if verbose:
            print("[HEAD] No cut needed.", flush=True)
        return df.copy().reset_index(drop=True)

    if n_keep <= 0:
        if verbose:
            print("[HEAD] Target ≤ 0, returning empty df.", flush=True)
        return df.iloc[0:0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    item_groups = df.groupby(item_col, sort=False)
    item_counts = item_groups.size().sort_values(ascending=False)
    max_count = int(item_counts.max())

    if verbose:
        print(
            f"[HEAD] items={len(item_counts):,} | max_count={max_count:,}",
            flush=True,
        )

    # Binary search over cap value
    selected_cap = max_count
    for cap in range(1, max_count + 1):
        total_keep = int(np.minimum(item_counts.values, cap).sum())
        if verbose and (cap == 1 or cap % cap_log_every == 0):
            print(
                f"[HEAD] cap={cap:,} → total_keep={total_keep:,} / {n_keep:,}",
                flush=True,
            )
        if total_keep >= n_keep:
            selected_cap = cap
            break

    cap = selected_cap

    # Keep cap-1 interactions for every item, then randomly add the remainder
    if cap > 1:
        keep_quota = item_counts.clip(upper=cap - 1).astype(int)
    else:
        keep_quota = pd.Series(0, index=item_counts.index, dtype=int)

    current_keep = int(keep_quota.sum())
    remaining_keep = n_keep - current_keep
    eligible_items = item_counts[item_counts >= cap].index.to_numpy()

    if remaining_keep > 0:
        chosen_items = rng.choice(eligible_items, size=remaining_keep, replace=False)
        keep_quota.loc[chosen_items] += 1

    item_indices = item_groups.indices
    keep_mask = np.zeros(len(df), dtype=bool)

    for item, quota in keep_quota.items():
        quota = int(quota)
        if quota <= 0:
            continue
        idx = item_indices[item]
        if quota >= len(idx):
            keep_mask[idx] = True
        else:
            keep_mask[rng.choice(idx, size=quota, replace=False)] = True

    result = df.iloc[keep_mask].copy().reset_index(drop=True)

    if verbose:
        print(
            f"[HEAD] Done | cap={cap:,} | kept={len(result):,} | "
            f"elapsed={perf_counter() - t_all:.2f}s",
            flush=True,
        )

    return result


def tail_item_cut(
    df: pd.DataFrame,
    item_col: str,
    keep_frac: float,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove all interactions of the rarest items (tail) until the target row
    count is reached.  The last item that would exceed the removal budget is
    partially removed via random sampling.

    No user/item protection — use thin_with_protected_user_item_interactions
    as the outer wrapper when protection is needed.
    """
    t_all = perf_counter()
    n_keep = int(round(len(df) * keep_frac))
    n_remove = len(df) - n_keep

    if verbose:
        print(
            f"[TAIL] keep_frac={keep_frac:.2f} | rows={len(df):,} | "
            f"target_keep={n_keep:,} | target_remove={n_remove:,}",
            flush=True,
        )

    if n_remove <= 0:
        if verbose:
            print("[TAIL] No cut needed.", flush=True)
        return df.copy().reset_index(drop=True)

    if n_keep <= 0:
        if verbose:
            print("[TAIL] Target keep ≤ 0, returning empty df.", flush=True)
        return df.iloc[0:0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)
    item_groups = df.groupby(item_col, sort=False)
    item_counts = item_groups.size().sort_values(ascending=True)
    item_indices = item_groups.indices

    if verbose:
        print(
            f"[TAIL] items={len(item_counts):,} | "
            f"min={int(item_counts.min()):,} | max={int(item_counts.max()):,}",
            flush=True,
        )

    keep_mask = np.ones(len(df), dtype=bool)
    remaining_remove = n_remove
    n_full_removed = 0
    n_partial_removed = 0

    for item, count in item_counts.items():
        if remaining_remove <= 0:
            break
        count = int(count)
        idx = item_indices[item]
        if count <= remaining_remove:
            keep_mask[idx] = False
            remaining_remove -= count
            n_full_removed += 1
        else:
            keep_mask[rng.choice(idx, size=remaining_remove, replace=False)] = False
            remaining_remove = 0
            n_partial_removed += 1

    result = df.iloc[keep_mask].copy().reset_index(drop=True)

    if verbose:
        print(
            f"[TAIL] Done | full_removed_items={n_full_removed:,} | "
            f"partial_removed_items={n_partial_removed:,} | "
            f"kept={len(result):,} | elapsed={perf_counter() - t_all:.2f}s",
            flush=True,
        )

    return result
