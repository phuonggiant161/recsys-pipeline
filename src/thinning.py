from pathlib import Path
import pandas as pd
import numpy as np
from time import perf_counter

#C1: Cut random
def random_thin_interactions(
    df: pd.DataFrame,
    keep_frac: float,
    seed: int = 42
) -> pd.DataFrame:
    """
    random giữ lại record theo tỉ lệ keep_fracs
    """
# làm tròn số dòng sau khi nhân với tỉ lệ keep_frac để cut
    n_keep = int(round(len(df) * keep_frac))

#random lấy số dòng
    thin_df = (
        df.sample(n=n_keep, random_state=seed, replace=False)
          .copy()
          .reset_index(drop=True)
    )

    return thin_df


def generate_random_thinning_levels(
    df: pd.DataFrame,
    keep_fracs: list[float],
    seed: int = 42
) -> dict[str, pd.DataFrame]:
    """
    gen nhiều dataset tương ứng với list fraction
    """

    outputs = {}

    for i, keep_frac in enumerate(keep_fracs, start=1):
        level_name = f"level_{i}_keep_{keep_frac:.2f}"
        outputs[level_name] = random_thin_interactions(
            df=df,
            keep_frac=keep_frac,
            seed=seed + i #mỗi level sẽ random khác nhau
        )

    return outputs


#C2: Cut top item bằng cách đặt trần cap
def head_item_cut(
    df: pd.DataFrame,
    item_col: str,
    keep_frac: float,
    seed: int = 42,
    verbose: bool = True,
    cap_log_every: int = 1000
) -> pd.DataFrame:
    """
    Cut top item bằng cách đặt trần số interaction tối đa cho mỗi item.

    Logic:
    - Tính số interaction cần giữ lại theo keep_frac.
    - Đếm số interaction của từng item.
    - Thử cap lần lượt từ 1, 2, 3,... cho đến khi đủ số dòng cần giữ.
    - Với mỗi item, số dòng được giữ tối đa là cap.
    - Nếu cap cuối cùng làm số dòng giữ lại bị vượt target,
      thì giữ cap - 1 trước, sau đó random thêm phần còn thiếu.
    """
    t_all = perf_counter()

    n_keep = int(round(len(df) * keep_frac))

    if verbose:
        print(
            f"[HEAD] Start | keep_frac={keep_frac:.2f} | "
            f"rows={len(df):,} | target_keep={n_keep:,}",
            flush=True
        )

    #nếu n_keep >= records df thì không cắt gì, còn nếu n_keep < 0 thì trả về rỗng
    if n_keep >= len(df):
        if verbose:
            print("[HEAD] No cut needed", flush=True)
        return df.copy().reset_index(drop=True)

    if n_keep <= 0:
        if verbose:
            print("[HEAD] Target keep <= 0, return empty df", flush=True)
        return df.iloc[0:0].copy().reset_index(drop=True)

    #tạo bộ random để random chọn item được cắt ở cap cuối
    rng = np.random.default_rng(seed)

    # đếm số interaction theo item
    t = perf_counter()

    item_counts = (
        df.groupby(item_col)
          .size()
          .sort_values(ascending=False)
    )

    max_count = int(item_counts.max())

    if verbose:
        print(
            f"[HEAD] Count item done | "
            f"items={len(item_counts):,} | max_count={max_count:,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # thử cap lần lượt từ 1, 2, 3,... cho đến khi đủ số dòng cần giữ
    t = perf_counter()

    selected_cap = max_count

    for cap in range(1, max_count + 1):
        total_keep = int(np.minimum(item_counts.values, cap).sum())

        if verbose and (cap == 1 or cap % cap_log_every == 0):
            print(
                f"[HEAD] Finding cap | cap={cap:,} | "
                f"total_keep={total_keep:,} / {n_keep:,}",
                flush=True
            )

        if total_keep >= n_keep:
            selected_cap = cap
            break

    cap = selected_cap

    if verbose:
        print(
            f"[HEAD] Find cap done | selected_cap={cap:,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # giữ chắc chắn cap - 1 interaction trước
    t = perf_counter()

    if cap > 1:
        keep_quota = item_counts.clip(upper=cap - 1).astype(int)
    else:
        keep_quota = pd.Series(0, index=item_counts.index, dtype=int)

    current_keep = int(keep_quota.sum())
    remaining_keep = n_keep - current_keep

    # các item có count >= cap là những item còn có thể giữ thêm 1 dòng
    eligible_items = item_counts[item_counts >= cap].index.to_numpy()

    if remaining_keep > 0:
        chosen_items = rng.choice(
            eligible_items,
            size=remaining_keep,
            replace=False
        )

        keep_quota.loc[chosen_items] += 1

    if verbose:
        print(
            f"[HEAD] Build keep quota done | "
            f"current_keep={current_keep:,} | remaining_keep={remaining_keep:,} | "
            f"final_keep={int(keep_quota.sum()):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # lấy index theo item một lần, không quét lại df cho từng item
    t = perf_counter()

    item_indices = df.groupby(item_col, sort=False).indices

    if verbose:
        print(
            f"[HEAD] Build item_indices done | "
            f"groups={len(item_indices):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # tạo mask để đánh dấu dòng nào được giữ
    t = perf_counter()

    keep_mask = np.zeros(len(df), dtype=bool)

    for item, quota in keep_quota.items():
        quota = int(quota)

        if quota <= 0:
            continue

        item_idx = item_indices[item]

        if quota >= len(item_idx):
            keep_mask[item_idx] = True
        else:
            chosen_idx = rng.choice(
                item_idx,
                size=quota,
                replace=False
            )

            keep_mask[chosen_idx] = True

    if verbose:
        print(
            f"[HEAD] Build keep mask done | "
            f"kept_rows={int(keep_mask.sum()):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # cắt dataframe một lần bằng mask
    t = perf_counter()

    thin_df = (
        df.iloc[keep_mask]
          .copy()
          .reset_index(drop=True)
    )

    if verbose:
        print(
            f"[HEAD] Slice df done | "
            f"thin_rows={len(thin_df):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

        print(
            f"[HEAD] Done | total_time={perf_counter() - t_all:.2f}s",
            flush=True
        )

    return thin_df


def generate_head_item_cut_levels(
    df: pd.DataFrame,
    item_col: str,
    keep_fracs: list[float],
    seed: int = 42,
    verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Gen nhiều dataset tương ứng với list keep_frac bằng cách top cut.
    """
    outputs = {}

    for i, keep_frac in enumerate(keep_fracs, start=1):
        level_name = f"top_item_keep_{keep_frac:.2f}"

        outputs[level_name] = head_item_cut(
            df=df,
            item_col=item_col,
            keep_frac=keep_frac,
            seed=seed + i
        )

    return outputs



#C3: Cut tail item
def tail_item_cut(
    df: pd.DataFrame,
    item_col: str,
    keep_frac: float,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Cut tail item bằng cách cắt sạch item ít interaction trước.

    Logic:
    - Tính số interaction cần giữ lại theo keep_frac.
    - Tính số interaction cần xóa.
    - Sắp xếp item theo số interaction tăng dần.
    - Cắt sạch tail item cho đến khi đủ số lượng cần xóa.
    - Nếu item cuối cùng làm vượt số lượng cần xóa,
      thì chỉ random xóa một phần interaction của item đó.
    """
    t_all = perf_counter()

    n_keep = int(round(len(df) * keep_frac))
    n_remove = len(df) - n_keep

    if verbose:
        print(
            f"[TAIL] Start | keep_frac={keep_frac:.2f} | "
            f"rows={len(df):,} | target_keep={n_keep:,} | "
            f"target_remove={n_remove:,}",
            flush=True
        )

    if n_remove <= 0:
        if verbose:
            print("[TAIL] No cut needed", flush=True)
        return df.copy().reset_index(drop=True)

    if n_keep <= 0:
        if verbose:
            print("[TAIL] Target keep <= 0, return empty df", flush=True)
        return df.iloc[0:0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # đếm số interaction theo item, item ít interaction đứng trước
    t = perf_counter()

    item_counts = (
        df.groupby(item_col)
          .size()
          .sort_values(ascending=True)
    )

    if verbose:
        print(
            f"[TAIL] Count item done | "
            f"items={len(item_counts):,} | "
            f"min_count={int(item_counts.min()):,} | "
            f"max_count={int(item_counts.max()):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # lấy index theo item một lần, không quét lại df cho từng item
    t = perf_counter()

    item_indices = df.groupby(item_col, sort=False).indices

    if verbose:
        print(
            f"[TAIL] Build item_indices done | "
            f"groups={len(item_indices):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # tạo mask ban đầu là giữ tất cả, sau đó đánh dấu False cho dòng bị xóa
    t = perf_counter()

    keep_mask = np.ones(len(df), dtype=bool)
    remaining_remove = n_remove

    n_full_removed_items = 0
    n_partial_removed_items = 0

    for item, count in item_counts.items():
        if remaining_remove <= 0:
            break

        item_idx = item_indices[item]
        count = int(count)

        # nếu cắt sạch item này vẫn chưa vượt số dòng cần xóa
        if count <= remaining_remove:
            keep_mask[item_idx] = False
            remaining_remove -= count
            n_full_removed_items += 1

        # nếu cắt sạch item này bị vượt thì chỉ xóa một phần
        else:
            chosen_idx = rng.choice(
                item_idx,
                size=remaining_remove,
                replace=False
            )

            keep_mask[chosen_idx] = False
            remaining_remove = 0
            n_partial_removed_items += 1

    if verbose:
        print(
            f"[TAIL] Build keep mask done | "
            f"full_removed_items={n_full_removed_items:,} | "
            f"partial_removed_items={n_partial_removed_items:,} | "
            f"removed_rows={len(df) - int(keep_mask.sum()):,} | "
            f"kept_rows={int(keep_mask.sum()):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

    # cắt dataframe một lần bằng mask
    t = perf_counter()

    thin_df = (
        df.iloc[keep_mask]
          .copy()
          .reset_index(drop=True)
    )

    if verbose:
        print(
            f"[TAIL] Slice df done | "
            f"thin_rows={len(thin_df):,} | "
            f"time={perf_counter() - t:.2f}s",
            flush=True
        )

        print(
            f"[TAIL] Done | total_time={perf_counter() - t_all:.2f}s",
            flush=True
        )

    return thin_df

def generate_tail_item_cut_levels(
    df: pd.DataFrame,
    item_col: str,
    keep_fracs: list[float],
    seed: int = 42,
    verbose: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Gen nhiều dataset tương ứng với list keep_frac bằng cách tail cut.
    """
    outputs = {}

    for i, keep_frac in enumerate(keep_fracs, start=1):
        level_name = f"tail_item_keep_{keep_frac:.2f}"

        outputs[level_name] = tail_item_cut(
            df=df,
            item_col=item_col,
            keep_frac=keep_frac,
            seed=seed + i
        )

    return outputs