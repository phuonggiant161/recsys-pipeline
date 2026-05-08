from pathlib import Path
import pandas as pd
import numpy as np

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
    seed: int = 42
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

    n_keep = int(round(len(df) * keep_frac))

    #nếu n_keep >= records df thì không cắt gì, còn nếu n_keep < 0 thì trả về rỗng
    if n_keep >= len(df):
        return df.copy().reset_index(drop=True)

    if n_keep <= 0:
        return df.iloc[0:0].copy().reset_index(drop=True)

    #tạo bộ random để random chọn item được cắt ở cap cuối
    rng = np.random.default_rng(seed)

    # đếm số interaction theo item
    item_counts = (
        df.groupby(item_col)
          .size()
          .sort_values(ascending=False)
    )
    # tìm max interaction của tất cả item để chạy loop từ 1 --> max
    max_count = int(item_counts.max())

    # thử cap lần lượt từ 1, 2, 3,... cho đến khi đủ số dòng cần giữ
    selected_cap = max_count

    for cap in range(1, max_count + 1):
        total_keep = int(np.minimum(item_counts.values, cap).sum())

        if total_keep >= n_keep:
            selected_cap = cap
            break

    cap = selected_cap

    # giữ chắc chắn cap - 1 interaction trước
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

        for item in chosen_items:
            keep_quota.loc[item] += 1

    keep_indices = []

    for item, quota in keep_quota.items():
        if quota <= 0:
            continue

        item_idx = df.index[df[item_col] == item].to_numpy()

        chosen_idx = rng.choice(
            item_idx,
            size=int(quota),
            replace=False
        )

        keep_indices.extend(chosen_idx.tolist())

    thin_df = (
        df.loc[sorted(keep_indices)]
          .copy()
          .reset_index(drop=True)
    )

    return thin_df


def generate_head_item_cut_levels(
    df: pd.DataFrame,
    item_col: str,
    keep_fracs: list[float],
    seed: int = 42
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
    seed: int = 42
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

    n_keep = int(round(len(df) * keep_frac))
    n_remove = len(df) - n_keep

    if n_remove <= 0:
        return df.copy().reset_index(drop=True)

    if n_keep <= 0:
        return df.iloc[0:0].copy().reset_index(drop=True)

    rng = np.random.default_rng(seed)

    # đếm số interaction theo item, item ít interaction đứng trước
    item_counts = (
        df.groupby(item_col)
          .size()
          .sort_values(ascending=True)
    )

    remove_indices = []
    remaining_remove = n_remove

    for item, count in item_counts.items():
        if remaining_remove <= 0:
            break

        item_idx = df.index[df[item_col] == item].to_numpy()

        # nếu cắt sạch item này vẫn chưa vượt số dòng cần xóa
        if count <= remaining_remove:
            remove_indices.extend(item_idx.tolist())
            remaining_remove -= int(count)

        # nếu cắt sạch item này bị vượt thì chỉ xóa một phần
        else:
            chosen_idx = rng.choice(
                item_idx,
                size=remaining_remove,
                replace=False
            )

            remove_indices.extend(chosen_idx.tolist())
            remaining_remove = 0

    thin_df = (
        df.drop(index=remove_indices)
          .copy()
          .reset_index(drop=True)
    )

    return thin_df


def generate_tail_item_cut_levels(
    df: pd.DataFrame,
    item_col: str,
    keep_fracs: list[float],
    seed: int = 42
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