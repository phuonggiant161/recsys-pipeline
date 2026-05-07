import math
import pandas as pd


def userwise_temporal_split(
    df: pd.DataFrame,
    user_col: str,
    timestamp_col: str,
    test_size: float = 0.2,
    min_train_interactions: int = 1,
    min_test_interactions: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
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

    work_df = (
        df.dropna(subset=[user_col, timestamp_col])
          .copy()
    )

    # chuyển timestamp về dạng datetime để split đúng theo thời gian
    work_df[timestamp_col] = pd.to_datetime(
        work_df[timestamp_col],
        errors="coerce"
    )

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

        # nếu user không đủ interaction để vừa có train vừa có test thì đưa toàn bộ vào train
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