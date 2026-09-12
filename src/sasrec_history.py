"""
Build SASRec `item_id_list` histories from already-fixed train/valid/test splits.

This module is intentionally decoupled from thinning/splitting: it must be
called *after* train/valid/test row membership has been finalized for a given
dataset variant (base or thinned). It never changes which rows belong to
train/valid/test -- it only computes each row's allowed history and writes it
into an `item_id_list` column.

Public API
----------
build_sasrec_histories(train_df, valid_df, test_df, user_col, item_col,
                        timestamp_col, row_seq_col, max_item_list_length,
                        fake_item_token)
    Returns (train_out, valid_out, test_out) -- copies of the inputs with
    `item_id_list` added/replaced and `row_seq_col` dropped.

History rules (advisor-confirmed evaluation protocol -- fully rolling by
chronological phase)
------------------------------------------------------------------------
  TRAIN target : history = that user's earlier TRAIN interactions only.
  VALID target : history = that user's earlier TRAIN + VALID interactions,
                 in chronological order (rolling). A VALID target is never
                 included in its OWN history, but once recorded it becomes
                 visible to a *later* VALID (and TEST) target from the same
                 user.
  TEST target   : history = that user's earlier TRAIN + VALID + TEST
                 interactions, in chronological order (rolling history).
                 A TEST target is never included in its OWN history, but
                 once its history has been recorded it is appended to the
                 pool so a *later* TEST target from the same user can see
                 it.

  Rolling is intentional, not leakage: at evaluation time RecBole only ever
  receives one row's own `item_id_list` (the earlier interactions), never
  the row's own target item, and a given target's history is fixed before
  the model ever sees it.

Ordering is by timestamp per user; exact timestamp ties are broken by
`row_seq_col`, a stable identifier assigned once (before any split or
thinning) so that ordering stays deterministic and reproducible.
"""

import pandas as pd

FAKE_ITEM_TOKEN = "__FAKE_ITEM__"
ROW_SEQ_COL = "__row_seq"


def _validate_columns(df: pd.DataFrame, name: str, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{name} is missing required columns: {sorted(missing)}")


def build_sasrec_histories(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_col: str,
    item_col: str,
    timestamp_col: str,
    row_seq_col: str = ROW_SEQ_COL,
    max_item_list_length: int = 50,
    fake_item_token: str = FAKE_ITEM_TOKEN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute SASRec `item_id_list` for a fixed train/valid/test split.

    Parameters
    ----------
    train_df, valid_df, test_df : Final split DataFrames (row membership must
        already be decided -- this function does not filter or reassign rows).
        Each must contain user_col, item_col, timestamp_col, row_seq_col.
    row_seq_col : Column holding a globally unique, stable identifier assigned
        once (e.g. right after k-core, before any split/thinning). Used only
        as a deterministic tie-break when two interactions share a timestamp.
    max_item_list_length : Keep at most this many most-recent allowed items.
    fake_item_token : Token used when the allowed history for a row is empty.

    Returns
    -------
    (train_out, valid_out, test_out) -- copies of the inputs with
    `item_id_list` set and row_seq_col dropped. Inputs are not mutated.
    """
    if max_item_list_length < 1:
        raise ValueError("max_item_list_length must be at least 1")

    required = {user_col, item_col, timestamp_col, row_seq_col}
    for name, df in [("train_df", train_df), ("valid_df", valid_df), ("test_df", test_df)]:
        _validate_columns(df, name, required)

    all_seqs = pd.concat(
        [train_df[row_seq_col], valid_df[row_seq_col], test_df[row_seq_col]]
    )
    if all_seqs.duplicated().any():
        raise ValueError(
            f"Duplicate {row_seq_col!r} values found across train/valid/test -- "
            "row_seq_col must be a globally unique identifier assigned before "
            "any split or thinning."
        )

    parts = []
    for split_name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        part = df[[user_col, item_col, timestamp_col, row_seq_col]].copy()
        part["__split"] = split_name
        part["__orig_index"] = df.index
        parts.append(part)
    work = pd.concat(parts, ignore_index=True)

    work["__parsed_ts"] = pd.to_datetime(work[timestamp_col], errors="coerce")
    n_bad = int(work["__parsed_ts"].isna().sum())
    if n_bad > 0:
        raise ValueError(
            f"{n_bad:,} row(s) have unparseable timestamps in column "
            f"'{timestamp_col}' while building SASRec histories."
        )

    work = work.sort_values(
        [user_col, "__parsed_ts", row_seq_col], kind="stable"
    )

    # (split, orig_index) -> history string. Keyed by split because each of
    # train_df/valid_df/test_df has its own independent 0..n-1 index.
    histories: dict[tuple[str, object], str] = {}

    for _, user_rows in work.groupby(user_col, sort=False):
        train_pool: list[str] = []        # TRAIN only -> read by TRAIN rows
        train_valid_pool: list[str] = []  # TRAIN + VALID -> read by VALID rows
        combined_pool: list[str] = []     # TRAIN + VALID + TEST -> read by TEST rows
        for split, orig_idx, item_id in zip(
            user_rows["__split"], user_rows["__orig_index"], user_rows[item_col]
        ):
            item_str = str(item_id)
            if split == "train":
                source = train_pool
            elif split == "valid":
                source = train_valid_pool
            else:
                source = combined_pool

            # 1. Choose the pool for this split, 2. build this row's history
            #    from *previously seen* interactions only, 3. save it --
            #    before touching any pool below.
            if source:
                histories[(split, orig_idx)] = " ".join(source[-max_item_list_length:])
            else:
                histories[(split, orig_idx)] = fake_item_token

            # 4. Only now append the current target, to every pool it should
            #    become visible in for a LATER row of the same user -- never
            #    its own. TRAIN rows must never leak into VALID/TEST history
            #    ordering assumptions, so each split feeds forward only into
            #    the pools that are allowed to see it:
            #      TRAIN -> train_pool, train_valid_pool, combined_pool
            #      VALID -> train_valid_pool, combined_pool
            #      TEST  -> combined_pool
            if split == "train":
                train_pool.append(item_str)
                train_valid_pool.append(item_str)
                combined_pool.append(item_str)
            elif split == "valid":
                train_valid_pool.append(item_str)
                combined_pool.append(item_str)
            else:
                combined_pool.append(item_str)

    def _assign(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        out = df.drop(columns=[row_seq_col]).copy()
        out["item_id_list"] = [histories[(split_name, idx)] for idx in df.index]
        return out

    return (
        _assign(train_df, "train"),
        _assign(valid_df, "valid"),
        _assign(test_df, "test"),
    )
