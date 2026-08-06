import heapq
import math
import numpy as np
import pandas as pd

from src.thinning import build_protected_row_ids


def userwise_temporal_holdout_with_coverage(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    holdout_size: float,
    split_name: str = "holdout",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal suffix split with protected user+item coverage.

    For each user u with n_u interactions:
        desired_u  = min(ceil(n_u * holdout_size), n_u - 1)
        window_u   = contiguous suffix after user u's last protected interaction
        actual_u   = min(desired_u, len(window_u))
        holdout_u  = latest actual_u rows of window_u

    No global row target, no heap, no cross-user timestamp comparison.
    Shortfall (desired_u > actual_u) is valid when protected rows block the
    suffix — it is not an error condition.

    Guarantees
    ----------
    * remainder contains every user and every item from df.
    * holdout users and items are subsets of remainder users and items.
    * holdout is a temporal suffix for every user represented in holdout.
    * len(remainder) + len(holdout) == len(df).
    * No user has zero interactions in remainder.
    * Each user's holdout count <= desired_u.
    """
    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size must be between 0 and 1.")

    work = df.reset_index(drop=True).copy()
    work["__row_id"] = np.arange(len(work), dtype=np.int64)

    # Fail-fast: unparseable timestamps would silently shrink the dataset.
    parsed_ts = pd.to_datetime(work[timestamp_col], errors="coerce")
    n_bad = int(parsed_ts.isna().sum())
    if n_bad > 0:
        raise ValueError(
            f"[{split_name}] {n_bad:,} row(s) have unparseable timestamps in "
            f"column '{timestamp_col}'. Fix the data before splitting."
        )
    work[timestamp_col] = parsed_ts

    protected_ids = build_protected_row_ids(work, user_col, item_col, timestamp_col)
    cols_out = df.columns.tolist()

    holdout_ids: list[int] = []
    n_users_with_holdout = 0
    n_users_single_interaction = 0
    n_users_limited_by_protected = 0
    desired_list: list[int] = []
    actual_list: list[int] = []

    for _, user_df in work.groupby(user_col, sort=False):
        user_sorted = user_df.sort_values(
            [timestamp_col, "__row_id"], kind="stable"
        )
        row_ids = user_sorted["__row_id"].tolist()
        n_u = len(row_ids)

        if n_u == 1:
            # Single-interaction users: desired = min(ceil(1*size), 0) = 0
            n_users_single_interaction += 1
            desired_list.append(0)
            actual_list.append(0)
            continue

        desired = min(math.ceil(n_u * holdout_size), n_u - 1)
        desired_list.append(desired)

        # Find last protected interaction in sorted order (scan from end)
        last_protected_pos = -1
        for i in range(n_u - 1, -1, -1):
            if row_ids[i] in protected_ids:
                last_protected_pos = i
                break

        # window = contiguous suffix after the last protected row
        window = row_ids[last_protected_pos + 1:]
        actual = min(desired, len(window))
        actual_list.append(actual)

        if actual < desired:
            # Protected suffix blocked some or all of the quota — valid outcome
            n_users_limited_by_protected += 1

        if actual > 0:
            holdout_ids.extend(window[-actual:])
            n_users_with_holdout += 1

    total_desired = sum(desired_list)
    total_actual  = sum(actual_list)
    shortfall     = total_desired - total_actual
    n_users_total = len(desired_list)
    n_users_without_holdout = n_users_total - n_users_with_holdout

    holdout_id_set = set(holdout_ids)
    holdout_mask   = work["__row_id"].isin(holdout_id_set)
    remainder_df   = work.loc[~holdout_mask, cols_out].reset_index(drop=True)
    holdout_df     = work.loc[holdout_mask,  cols_out].reset_index(drop=True)

    actual_overall_ratio = len(holdout_df) / len(df) if len(df) > 0 else 0.0

    # ── Diagnostics ───────────────────────────────────────────────────────────
    desired_arr = np.array(desired_list)
    actual_arr  = np.array(actual_list)
    nonzero_desired = desired_arr[desired_arr > 0]
    nonzero_actual  = actual_arr[actual_arr > 0]

    print(
        f"  [{split_name}] "
        f"users={n_users_total:,} | "
        f"users_with_holdout={n_users_with_holdout:,} | "
        f"users_without_holdout={n_users_without_holdout:,} | "
        f"users_single_interaction={n_users_single_interaction:,} | "
        f"users_limited_by_protected={n_users_limited_by_protected:,}"
    )
    print(
        f"  [{split_name}] "
        f"desired_sum={total_desired:,} | "
        f"actual_sum={total_actual:,} | "
        f"shortfall={shortfall:,} | "
        f"actual_overall_ratio={actual_overall_ratio:.4f}"
    )
    if len(nonzero_desired) > 0:
        print(
            f"  [{split_name}] desired per user (nonzero): "
            f"min={int(nonzero_desired.min())} "
            f"median={int(np.median(nonzero_desired))} "
            f"max={int(nonzero_desired.max())}"
        )
    if len(nonzero_actual) > 0:
        print(
            f"  [{split_name}] actual  per user (nonzero): "
            f"min={int(nonzero_actual.min())} "
            f"median={int(np.median(nonzero_actual))} "
            f"max={int(nonzero_actual.max())}"
        )

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(remainder_df) + len(holdout_df) == len(df), (
        f"BUG [{split_name}]: row count mismatch — "
        f"remainder={len(remainder_df)} + holdout={len(holdout_df)} "
        f"!= input={len(df)}"
    )
    remainder_row_ids = set(work.loc[~holdout_mask, "__row_id"])
    assert holdout_id_set.isdisjoint(remainder_row_ids), (
        f"BUG [{split_name}]: overlap between remainder and holdout row IDs"
    )
    missing_u = set(df[user_col]) - set(remainder_df[user_col])
    assert not missing_u, (
        f"BUG [{split_name}]: remainder missing users: {missing_u}"
    )
    missing_i = set(df[item_col]) - set(remainder_df[item_col])
    assert not missing_i, (
        f"BUG [{split_name}]: remainder missing items: {missing_i}"
    )
    if len(holdout_df) > 0:
        extra_u = set(holdout_df[user_col]) - set(remainder_df[user_col])
        assert not extra_u, (
            f"BUG [{split_name}]: holdout has users not in remainder: {extra_u}"
        )
        extra_i = set(holdout_df[item_col]) - set(remainder_df[item_col])
        assert not extra_i, (
            f"BUG [{split_name}]: holdout has items not in remainder: {extra_i}"
        )
        h_min = holdout_df.groupby(user_col)[timestamp_col].min()
        r_max = remainder_df.groupby(user_col)[timestamp_col].max()
        for u, min_h in h_min.items():
            max_r = r_max.get(u)
            if max_r is not None:
                assert max_r <= min_h, (
                    f"BUG [{split_name}]: temporal violation for user {u}: "
                    f"max(remainder)={max_r} > min(holdout)={min_h}"
                )
        input_counts = work.groupby(user_col, sort=False).size()
        holdout_counts = (
            holdout_df.groupby(user_col).size()
            .reindex(input_counts.index, fill_value=0)
        )
        desired_counts = np.minimum(
            np.ceil(input_counts * holdout_size).astype(int),
            (input_counts - 1).clip(lower=0),
        )
        violations = holdout_counts[holdout_counts > desired_counts]
        assert violations.empty, (
            f"BUG [{split_name}]: holdout quota exceeded for users: "
            f"{violations.to_dict()}"
        )

    return remainder_df, holdout_df


def temporal_global_holdout_with_coverage(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    holdout_size: float,
    protect_user: bool = True,
    protect_item: bool = True,
    split_name: str = "holdout",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED — not used by the current preprocessing pipeline.

    The current pipeline uses userwise_temporal_holdout_with_coverage() instead,
    which applies per-user ceil() quotas and no global row target.  This function
    uses a global target (round(N * holdout_size)) and a max-heap to enforce it;
    behaviour differs substantially from the per-user rule.  Kept for reference
    and backward compatibility only.

    Original docstring below.
    --------------------------
    Global temporal holdout split with coverage protection.

    Selects  target = round(len(df) * holdout_size)  rows for holdout using a
    greedy newest-first scan:

      1. Each user's "frontier" is their latest unvisited interaction.
      2. All frontiers are maintained in a max-heap by (timestamp, row_id).
      3. For each popped candidate, accept it into holdout only if:
           (a) the user still has >= 1 row remaining in the non-holdout part
               (when protect_user=True)
           (b) the item still has >= 1 row remaining in the non-holdout part
               (when protect_item=True)
      4. After a row is accepted, advance that user's frontier (next-newest row)
         and push it onto the heap.
      5. If a row is REJECTED (coverage would be broken), the user is exhausted
         entirely — older rows of that user are NOT considered.  This preserves
         the suffix property: it is wrong to hold out an older interaction while
         a newer one from the same user stays in the remainder.
      6. Stop when target rows are collected or the heap is empty.

    Guarantees (when both protect flags are True)
    ---------------------------------------------
    * remainder contains every user and every item from df.
    * holdout is a temporal suffix for every user represented in holdout.
    * No user with exactly 1 interaction contributes to holdout.
    * len(remainder) + len(holdout) == len(df)  (no rows lost).

    If the target cannot be fully reached, a WARNING is printed with the
    shortfall and reason (coverage or temporal constraint).

    Parameters
    ----------
    df            : Input DataFrame.
    user_col      : Column name for user IDs.
    item_col      : Column name for item IDs.
    timestamp_col : Column name for timestamps (any format pd.to_datetime accepts).
    holdout_size  : Fraction of rows to put in holdout  (0 < holdout_size < 1).
    protect_user  : Ensure every user stays in remainder.
    protect_item  : Ensure every item stays in remainder.
    split_name    : Label used in WARNING / assertion messages.

    Returns
    -------
    (remainder_df, holdout_df)  — both have the same columns as df.
    """
    if not 0 < holdout_size < 1:
        raise ValueError("holdout_size must be between 0 and 1.")

    work = df.reset_index(drop=True).copy()
    work["__row_id"] = np.arange(len(work), dtype=np.int64)

    # Fail-fast on invalid timestamps — silent drop would shrink the dataset
    # and break the len(remainder)+len(holdout)==len(df) guarantee.
    parsed_ts = pd.to_datetime(work[timestamp_col], errors="coerce")
    n_bad = int(parsed_ts.isna().sum())
    if n_bad > 0:
        raise ValueError(
            f"[{split_name}] {n_bad:,} row(s) have unparseable timestamps in "
            f"column '{timestamp_col}'. Fix the data before splitting."
        )
    work[timestamp_col] = parsed_ts
    n = len(work)

    target = max(0, round(n * holdout_size))
    cols_out = df.columns.tolist()

    if target == 0:
        return (
            work[cols_out].reset_index(drop=True),
            work.iloc[0:0][cols_out].reset_index(drop=True),
        )

    # ── Precompute fast lookup arrays ─────────────────────────────────────────
    rid_arr   = work["__row_id"].values
    user_arr  = work[user_col].values
    item_arr  = work[item_col].values
    # Convert timestamps to int64 nanoseconds for heap ordering
    ts_ns_arr = work[timestamp_col].values.astype(np.int64)

    rid_to_idx: dict[int, int] = {int(rid): i for i, rid in enumerate(rid_arr)}

    # ── Per-user sorted sequences: oldest → newest (list of row_ids) ─────────
    user_seq: dict = {}
    for user, grp in work.groupby(user_col, sort=False):
        sorted_rids = (
            grp.sort_values([timestamp_col, "__row_id"], kind="stable")["__row_id"]
            .values.astype(int)
            .tolist()
        )
        user_seq[user] = sorted_rids

    # ── Per-user frontier: index into user_seq (starts at last = newest) ─────
    user_frontier: dict = {u: len(seq) - 1 for u, seq in user_seq.items()}

    # ── Train counts: start at total occurrences per user / item ─────────────
    train_user_count: dict = {u: len(seq) for u, seq in user_seq.items()}
    train_item_count: dict = {}
    for item, grp in work.groupby(item_col, sort=False):
        train_item_count[item] = len(grp)

    # ── Max-heap: (-ts_ns, -row_id, user) — row_id is globally unique so
    #    (-ts_ns, -row_id) is always a unique key; user never used in comparison.
    heap: list = []
    for user, seq in user_seq.items():
        rid = seq[-1]
        ts_ns = int(ts_ns_arr[rid_to_idx[rid]])
        heapq.heappush(heap, (-ts_ns, -rid, user))

    holdout_ids: set[int] = set()

    # ── Greedy selection ──────────────────────────────────────────────────────
    while heap and len(holdout_ids) < target:
        neg_ts, neg_rid, user = heapq.heappop(heap)
        rid = int(-neg_rid)

        fidx = user_frontier[user]
        if fidx < 0:
            continue  # user already exhausted (stale heap entry)
        if user_seq[user][fidx] != rid:
            continue  # stale entry (shouldn't happen, guard anyway)

        idx  = rid_to_idx[rid]
        item = item_arr[idx]

        # Coverage check
        can_take = True
        if protect_user and train_user_count[user] <= 1:
            can_take = False
        if can_take and protect_item and train_item_count[item] <= 1:
            can_take = False

        if can_take:
            holdout_ids.add(rid)
            train_user_count[user] -= 1
            train_item_count[item] -= 1
            user_frontier[user] -= 1
            new_fidx = user_frontier[user]
            if new_fidx >= 0:
                next_rid = user_seq[user][new_fidx]
                next_ts_ns = int(ts_ns_arr[rid_to_idx[next_rid]])
                heapq.heappush(heap, (-next_ts_ns, -next_rid, user))
        else:
            # Suffix property: if latest row can't go to holdout, skip user entirely
            user_frontier[user] = -1

    # ── Build output DataFrames ───────────────────────────────────────────────
    holdout_mask = work["__row_id"].isin(holdout_ids)
    remainder_df = work.loc[~holdout_mask, cols_out].reset_index(drop=True)
    holdout_df   = work.loc[holdout_mask,  cols_out].reset_index(drop=True)

    # ── Log warning if target not fully met ───────────────────────────────────
    actual = len(holdout_df)
    if actual != target:
        short = target - actual
        print(
            f"  [{split_name} WARNING] target={target:,} actual={actual:,} "
            f"short={short:,} — coverage or temporal constraint prevented full target"
        )

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(remainder_df) + len(holdout_df) == len(df), (
        f"BUG [{split_name}]: row count mismatch — "
        f"remainder={len(remainder_df)} + holdout={len(holdout_df)} "
        f"!= input={len(df)}"
    )
    if protect_user:
        missing_u = set(df[user_col]) - set(remainder_df[user_col])
        assert not missing_u, (
            f"BUG [{split_name}]: remainder missing users from input: {missing_u}"
        )
    if protect_item:
        missing_i = set(df[item_col]) - set(remainder_df[item_col])
        assert not missing_i, (
            f"BUG [{split_name}]: remainder missing items from input: {missing_i}"
        )
    # Temporal suffix check: for every user in holdout, max(remainder_ts) <= min(holdout_ts)
    if len(holdout_df) > 0:
        h_min = holdout_df.groupby(user_col)[timestamp_col].min()
        r_max = remainder_df.groupby(user_col)[timestamp_col].max()
        for u, min_h in h_min.items():
            max_r = r_max.get(u)
            if max_r is not None:
                assert max_r <= min_h, (
                    f"BUG [{split_name}]: temporal violation for user {u}: "
                    f"max(remainder)={max_r} > min(holdout)={min_h}"
                )

    return remainder_df, holdout_df


def userwise_temporal_split(
    df: pd.DataFrame,
    user_col: str,
    timestamp_col: str,
    test_size: float = 0.2,
    min_train_interactions: int = 1,
    min_test_interactions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED — not used by the current preprocessing pipeline.

    Does not enforce user+item coverage in the remainder; use
    userwise_temporal_holdout_with_coverage() instead.

    Chia train/test theo từng user dựa trên timestamp.

    Với mỗi user:
    - sort interaction theo thời gian tăng dần
    - phần cũ hơn đưa vào train
    - phần mới hơn đưa vào test
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    required_cols = [user_col, timestamp_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataframe: {missing_cols}")

    train_parts = []
    test_parts = []
    skipped_users = 0

    work_df = df.dropna(subset=[user_col, timestamp_col]).copy()
    work_df[timestamp_col] = pd.to_datetime(work_df[timestamp_col], errors="coerce")
    work_df = (
        work_df.dropna(subset=[timestamp_col])
               .sort_values(
                   by=[user_col, timestamp_col],
                   ascending=[True, True],
                   kind="mergesort"
               )
    )

    for _, user_df in work_df.groupby(user_col, sort=False):
        n = len(user_df)
        if n < min_train_interactions + min_test_interactions:
            train_parts.append(user_df)
            skipped_users += 1
            continue

        n_test = int(math.ceil(n * test_size))
        n_test = max(n_test, min_test_interactions)
        n_train = n - n_test

        if n_train < min_train_interactions:
            n_train = min_train_interactions
            n_test = n - n_train

        train_parts.append(user_df.iloc[:n_train])
        test_parts.append(user_df.iloc[n_train:])

    train_df = (
        pd.concat(train_parts, ignore_index=True)
        if train_parts
        else pd.DataFrame(columns=df.columns)
    )
    test_df = (
        pd.concat(test_parts, ignore_index=True)
        if test_parts
        else pd.DataFrame(columns=df.columns)
    )

    print(f"Users not split due to insufficient interactions: {skipped_users}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def split_train_valid_preserve_items(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    valid_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    DEPRECATED — not used by the current preprocessing pipeline.

    Uses a global validation target distributed via Largest Remainder Method.
    The current pipeline uses userwise_temporal_holdout_with_coverage() with
    per-user ceil() quotas and no global target.  Kept for reference only.

    Split df into train/valid with three guarantees:
      1. train contains every user and every item from df
      2. valid users/items are a subset of train
      3. temporal: max(train_ts_user) <= min(valid_ts_user) for all users with valid rows

    Valid-size allocation
    --------------------
    Global target:  target_valid = round(len(df) * valid_ratio)

    Per-user "valid window" = trailing non-protected interactions (scan from end,
    stop at first protected row).  Protected rows always go to train.

    The target_valid rows are distributed across users' windows using the
    Largest Remainder Method (proportional allocation), so the aggregate
    valid ratio matches valid_ratio regardless of each user's window size.
    This avoids the per-user ceil() inflation that occurs for sparse users.

    If protected interactions constrain available slots below target_valid,
    a warning is printed and all available slots are used.

    Raises
    ------
    AssertionError  if any guarantee is violated (always indicates a bug).
    """
    if not 0 < valid_ratio < 1:
        raise ValueError("valid_ratio must be between 0 and 1.")

    work = df.reset_index(drop=True).copy()
    work["__row_id"] = np.arange(len(work), dtype=np.int64)
    work[timestamp_col] = pd.to_datetime(work[timestamp_col], errors="coerce")
    work = work.dropna(subset=[timestamp_col]).reset_index(drop=True)
    work["__row_id"] = np.arange(len(work), dtype=np.int64)

    protected_ids = build_protected_row_ids(work, user_col, item_col, timestamp_col)
    n_protected = len(protected_ids)

    # ── Per-user: collect valid window row_ids (oldest → newest within window) ──
    user_window_ids: dict[object, list[int]] = {}

    for user, user_df in work.groupby(user_col, sort=False):
        user_sorted = user_df.sort_values([timestamp_col, "__row_id"], kind="stable")
        row_ids = user_sorted["__row_id"].tolist()
        n = len(row_ids)

        window_size = 0
        for i in range(n - 1, -1, -1):
            if row_ids[i] in protected_ids:
                break
            window_size += 1

        if window_size > 0:
            # Last window_size elements (most recent end of user's timeline)
            user_window_ids[user] = row_ids[n - window_size:]

    total_available = sum(len(w) for w in user_window_ids.values())
    target_valid = max(0, round(len(work) * valid_ratio))

    # ── Distribute target_valid across user windows ────────────────────────────
    valid_row_ids: set[int] = set()

    if target_valid == 0 or total_available == 0:
        # Nothing to put in valid
        pass

    elif total_available <= target_valid:
        # Protected rows are so many that valid window can't reach target.
        # Take everything available and log a warning.
        for window in user_window_ids.values():
            valid_row_ids.update(window)
        print(
            f"  [valid_split WARNING] target_valid={target_valid:,}, "
            f"total_available={total_available:,}, protected={n_protected:,}. "
            f"Cannot reach target — taking all available valid rows."
        )

    else:
        # Largest Remainder Method: floor_alloc + top-up by fractional remainder
        exact = {u: len(w) * valid_ratio for u, w in user_window_ids.items()}
        floor_alloc: dict[object, int] = {u: int(math.floor(v)) for u, v in exact.items()}
        still_needed = target_valid - sum(floor_alloc.values())

        if still_needed > 0:
            # Sort by (fractional remainder desc, window_size desc) for stable top-up
            fractions = {
                u: (exact[u] - floor_alloc[u], len(user_window_ids[u]))
                for u in user_window_ids
            }
            top_users = sorted(fractions, key=fractions.__getitem__, reverse=True)
            for u in top_users[:still_needed]:
                floor_alloc[u] = min(floor_alloc[u] + 1, len(user_window_ids[u]))

        for u, n_alloc in floor_alloc.items():
            if n_alloc > 0:
                # Take the n_alloc most-recent rows from the user's window
                valid_row_ids.update(user_window_ids[u][-n_alloc:])

    # ── Build output DataFrames ───────────────────────────────────────────────
    cols_out = df.columns.tolist()
    valid_mask = work["__row_id"].isin(valid_row_ids)
    train_df = work.loc[~valid_mask, cols_out].reset_index(drop=True)
    valid_df = work.loc[valid_mask,  cols_out].reset_index(drop=True)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert len(train_df) + len(valid_df) == len(work), (
        f"BUG: rows lost during split: "
        f"{len(train_df)} + {len(valid_df)} != {len(work)}"
    )

    missing_users = set(df[user_col]) - set(train_df[user_col])
    assert not missing_users, (
        f"BUG: user set not fully preserved in train. Missing: {missing_users}"
    )
    missing_items = set(df[item_col]) - set(train_df[item_col])
    assert not missing_items, (
        f"BUG: item set not fully preserved in train. Missing: {missing_items}"
    )

    if len(valid_df) > 0:
        extra_users = set(valid_df[user_col]) - set(train_df[user_col])
        assert not extra_users, (
            f"BUG: valid contains users not in train: {extra_users}"
        )
        extra_items = set(valid_df[item_col]) - set(train_df[item_col])
        assert not extra_items, (
            f"BUG: valid contains items not in train: {extra_items}"
        )
        valid_user_min_ts = valid_df.groupby(user_col)[timestamp_col].min()
        train_user_max_ts = train_df.groupby(user_col)[timestamp_col].max()
        for user, min_valid_ts in valid_user_min_ts.items():
            max_train_ts = train_user_max_ts.get(user)
            if max_train_ts is not None:
                assert max_train_ts <= min_valid_ts, (
                    f"BUG: temporal violation for user {user}: "
                    f"max(train_ts)={max_train_ts} > min(valid_ts)={min_valid_ts}"
                )

    # ── Valid ratio diagnostic (warn if >0.1% off target) ────────────────────
    actual_ratio = len(valid_df) / len(work) if len(work) > 0 else 0
    if abs(actual_ratio - valid_ratio) > 0.001:
        print(
            f"  [valid_ratio] target={valid_ratio:.3f} actual={actual_ratio:.4f} "
            f"(diff={actual_ratio - valid_ratio:+.4f}) | "
            f"target_valid={target_valid:,} actual_valid={len(valid_df):,} | "
            f"total_available={total_available:,} protected={n_protected:,}"
        )

    return train_df, valid_df


def split_train_valid_from_train_pool(
    train_pool: pd.DataFrame,
    user_col: str,
    timestamp_col: str,
    valid_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """DEPRECATED — wrapper around the deprecated userwise_temporal_split."""
    return userwise_temporal_split(
        df=train_pool,
        user_col=user_col,
        timestamp_col=timestamp_col,
        test_size=valid_size,
        min_train_interactions=1,
        min_test_interactions=1,
    )
