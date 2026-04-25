from pathlib import Path
import pandas as pd


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


def tail_item_cut(
    df: pd.DataFrame,
    item_col: str,
    keep_frac: float
) -> pd.DataFrame:
    """
    Cắt bỏ một phần tail item (ít phổ biến) khỏi dataset
    Xếp thứ tự item có interaction ít nhất đến nhiều nhất, sau đó cắt bỏ phần tail theo tỉ lệ cut_frac.
    Cut frac trong khoảng [0,1]
    """

    n_keep = int(round(len(df) * keep_frac))
    n_remove = len(df) - n_keep
#group by item đếm interaction
    item_counts = (
        df.groupby(item_col)
          .size()
          .rename("item_interaction_count")
          .reset_index()
    )
#join lại với df gốc để có cột đếm số interaction của item, sau đó sắp xếp theo số interaction tăng dần, item ít phổ biến sẽ ở đầu, sau đó cắt bỏ phần đầu tương ứng với n_remove
    ranked_df = (
        df.merge(item_counts, on=item_col, how="left")
          .sort_values(
              by=["item_interaction_count", item_col],
              ascending=[True, True]
          )
          .reset_index(drop=True)
    )

    thin_df = (
        ranked_df.iloc[n_remove:]
                 .drop(columns=["item_interaction_count"])
                 .copy()
                 .reset_index(drop=True)
    )

    return thin_df


def generate_tail_item_cut_levels(
    df: pd.DataFrame,
    item_col: str,
    keep_fracs: list[float]
) -> dict[str, pd.DataFrame]:
    """Gen nhiều dataset tương ứng với list cut_fracs, mỗi dataset sẽ cắt bỏ phần tail item theo tỉ lệ cut_fracs"""

    outputs = {}

    for keep_frac in keep_fracs:
        level_name = f"tail_item_keep_{keep_frac:.2f}"

        outputs[level_name] = tail_item_cut(
            df=df,
            item_col=item_col,
            keep_frac=keep_frac
        )

    return outputs