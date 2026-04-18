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