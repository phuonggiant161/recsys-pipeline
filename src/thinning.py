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

#C2: cut có kiểm soát

def head_item_cut(
    df: pd.DataFrame,
    item_col: str,
    keep_frac: float,
    cut_ratio_per_round: float = 0.3,
    top_n_per_round: int = 100,
    seed: int = 42
) -> pd.DataFrame:
    """
    Cắt dữ liệu theo hướng kiểm soát distribution bằng cách ưu tiên cắt nhóm head item.

    Logic:
    - Tính số interaction cần giữ theo keep_frac.
    - Trong mỗi vòng:
        1. Đếm lại số interaction của từng item trên dữ liệu còn lại.
        2. Rank item theo số interaction giảm dần.
        3. Chọn top_n_per_round item phổ biến nhất hiện tại.
        4. Với mỗi item trong nhóm head, cắt cut_ratio_per_round interaction.
        5. Rank lại trên dữ liệu còn lại và tiếp tục lặp.
    - Dừng khi số interaction còn lại đạt mức cần giữ.

    Ví dụ:
    keep_frac = 0.1 nghĩa là giữ lại khoảng 10% interaction ban đầu.
    cut_ratio_per_round = 0.5 nghĩa là mỗi vòng cắt 50% interaction
    của từng item trong nhóm head hiện tại.
    top_n_per_round = 500 nghĩa là mỗi vòng xử lý 500 item phổ biến nhất.

    Lưu ý:
    - Đây là phiên bản batch của head thinning.
    - Thay vì mỗi vòng chỉ cắt 1 item top, hàm này cắt nhiều head item cùng lúc
      để giảm số vòng lặp trên dataset lớn.
    """


    # Số interaction ban đầu.
    n_original = len(df)

    # Số interaction cần giữ lại sau khi cắt.

    n_keep = int(round(n_original * keep_frac))

    # Tổng số interaction cần loại bỏ.
    n_remove_total = n_original - n_keep

    # Nếu không cần cắt thì trả về bản copy của dữ liệu gốc.
    if n_remove_total <= 0:
        return df.copy().reset_index(drop=True)

    # Khởi tạo random generator để việc chọn dòng cần xóa có thể tái lập.
    # Cùng seed thì kết quả cắt sẽ giống nhau.
    rng = np.random.default_rng(seed)

    # remaining_df là dữ liệu sẽ bị cắt dần qua từng vòng.
    remaining_df = df.copy().reset_index(drop=True)

    # Theo dõi tổng số dòng đã bị loại bỏ.
    removed_count = 0

    # Lặp đến khi đã cắt đủ số dòng cần cắt
    # hoặc số dòng còn lại đã bằng số dòng cần giữ.
    while removed_count < n_remove_total and len(remaining_df) > n_keep:

        # Đếm lại số interaction theo từng item trên dữ liệu hiện tại.
        # Sau mỗi vòng cắt, popularity của item có thể thay đổi,
        # nên cần groupby lại.
        item_counts = (
            remaining_df.groupby(item_col)
            .size()
            .sort_values(ascending=False)
        )

        # Nếu không còn item nào thì dừng.
        if item_counts.empty:
            break

        # Chọn nhóm item phổ biến nhất hiện tại.
        head_items = item_counts.head(top_n_per_round)

        # Số dòng còn cần cắt trong toàn bộ quá trình.
        remaining_to_remove = n_remove_total - removed_count

        # Danh sách index các dòng sẽ bị xóa trong vòng hiện tại.
        remove_indices = []

        # Duyệt qua từng item trong nhóm head item.
        for item, count in head_items.items():

            # Nếu đã đủ số dòng cần cắt thì dừng vòng for.
            if remaining_to_remove <= 0:
                break

            # Lấy index của tất cả interaction thuộc item hiện tại.
            item_idx = remaining_df.index[
                remaining_df[item_col] == item
            ].to_numpy()

            # Tính số interaction sẽ cắt khỏi item này.
            # Ví dụ item có 8,000 interaction,
            # cut_ratio_per_round = 0.5
            # thì cắt khoảng 4,000 interaction của item đó.
            n_remove_this_item = int(round(count * cut_ratio_per_round))

            # Đảm bảo mỗi item được chọn sẽ bị cắt ít nhất 1 dòng.
            # Nếu không có dòng này, khi count nhỏ, round có thể ra 0
            # và vòng lặp có thể không tiến triển.
            n_remove_this_item = max(1, n_remove_this_item)

            # Không cắt vượt quá tổng số dòng còn cần cắt.
            n_remove_this_item = min(
                n_remove_this_item,
                remaining_to_remove
            )

            # Không cắt vượt quá số interaction hiện có của item.
            n_remove_this_item = min(
                n_remove_this_item,
                len(item_idx)
            )

            # Random chọn các interaction của item này để xóa.
            # Không dùng item_idx[:n] để tránh phụ thuộc vào thứ tự dòng ban đầu,
            # đặc biệt nếu dữ liệu đang được sắp theo thời gian.
            chosen_idx = rng.choice(
                item_idx,
                size=n_remove_this_item,
                replace=False
            )

            # Thêm index vừa chọn vào danh sách xóa của vòng hiện tại.
            remove_indices.extend(chosen_idx.tolist())

            # Cập nhật số dòng còn cần xóa.
            remaining_to_remove -= n_remove_this_item

        # Nếu vì lý do nào đó không chọn được dòng nào để xóa thì dừng,
        # tránh vòng lặp vô hạn.
        if len(remove_indices) == 0:
            break

        # Xóa toàn bộ các dòng đã chọn trong vòng hiện tại.
        # Reset index để dataframe gọn lại cho vòng sau.
        remaining_df = (
            remaining_df.drop(index=remove_indices)
            .reset_index(drop=True)
        )

        # Cập nhật tổng số dòng đã xóa.
        removed_count += len(remove_indices)

    # Trả về dữ liệu sau khi thinning.
    return remaining_df.reset_index(drop=True)


def generate_head_item_cut_levels(
    df: pd.DataFrame,
    item_col: str,
    keep_fracs: list[float],
    cut_ratio_per_round: float = 0.3,
    top_n_per_round: int = 500,
    seed: int = 42
) -> dict[str, pd.DataFrame]:
    """
    Gen nhiều dataset theo nhiều mức keep_frac bằng phương pháp:
    cắt nhóm item head theo vòng lặp, sau mỗi vòng rank lại item.
    """

    outputs = {}

    for i, keep_frac in enumerate(keep_fracs, start=1):
        level_name = f"iter_head_item_keep_{keep_frac:.2f}"

        outputs[level_name] = head_item_cut(
            df=df,
            item_col=item_col,
            keep_frac=keep_frac,
            cut_ratio_per_round=cut_ratio_per_round,
            top_n_per_round=top_n_per_round,
            seed=seed + i
        )

    return outputs