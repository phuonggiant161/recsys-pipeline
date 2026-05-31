# recsys-pipeline

Pipeline nghiên cứu ảnh hưởng của data sparsity đến chất lượng recommender system.

**Luồng chính:**
```
Raw data → k-core filtering → Train/Test split → Thinning → Elliot training & evaluation
```

---

## Cấu trúc repo

```
recsys-pipeline/
├── main.py                        # Pipeline chính (kcore → split → thinning)
├── src/
│   ├── config.py                  # Khai báo dataset (input path, cột)
│   ├── thinning.py                # 3 chiến lược thinning: random, head-item, tail-item
│   ├── dataset_folder.py          # Lưu train/test + tự export .tsv cho Elliot
│   └── ...
├── scripts/
│   ├── generate_elliot_configs.py # Tự sinh Elliot config từ data/processed/
│   └── run_elliot.py              # Chạy Elliot experiments từ repo root
├── external/elliot/               # Elliot framework (submodule)
├── data/
│   ├── raw/                       # Dữ liệu thô (không commit)
│   ├── processed/                 # Output của main.py (không commit)
│   └── reports/                   # Sparsity summary CSV (commit)
└── results/elliot/
    └── performance/               # Kết quả metrics của Elliot (commit)
```

---

## Chuẩn bị dữ liệu

File dữ liệu thô không được commit do kích thước lớn. Tải về tại:

https://drive.google.com/drive/folders/15D99fn2hSeKwfRGJWadF9hDNB4DaRqrb

Sau khi tải về, đặt vào:
```
data/raw/hm.csv
data/raw/baby_product.parquet
```

---

## Cài đặt

### 1. Pipeline chính (preprocessing + thinning)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Elliot (training + evaluation)

Elliot yêu cầu Python environment riêng do dùng các thư viện cũ (numpy 1.18, tensorflow 2.3):

```bash
cd external/elliot
python -m venv venv_elliot
venv_elliot\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## Chạy pipeline

### Bước 1 — Preprocessing và thinning

Cấu hình tham số trong `main.py` (đầu file):

```python
DATASET_NAME = "hm"   # hoặc "baby_product"
K = 20                # k-core
KEEP_FRACS = [0.9, 0.7, 0.5, 0.3, 0.1]
```

```bash
python main.py
```

Output: `data/processed/hm_k20_*/` — mỗi folder chứa `train.csv`, `test.csv`, `train.tsv`, `test.tsv`, `metadata.json`.

### Bước 2 — Sinh Elliot config

```bash
# Sinh config cho tất cả dataset trong data/processed/
python scripts/generate_elliot_configs.py

# Chỉ sinh cho một subset
python scripts/generate_elliot_configs.py --filter hm_k20

# Ghi đè config đã có
python scripts/generate_elliot_configs.py --overwrite
```

Config được lưu vào `external/elliot/config_files/`.

### Bước 3 — Chạy Elliot

```bash
# Chạy tất cả experiments
python scripts/run_elliot.py

# Chỉ chạy một subset
python scripts/run_elliot.py --filter hm_k20_random

# Chạy experiment cụ thể
python scripts/run_elliot.py --config hm_k20_dedup_base_split_itemknn

# Xem command sẽ chạy mà không thực thi
python scripts/run_elliot.py --filter hm_k20 --dry-run
```

Kết quả ghi vào `results/elliot/performance/`.

---

## Metrics

Mỗi experiment đánh giá các metrics sau:

| Metric | Mô tả |
|--------|-------|
| `Precision` | Precision@k trên toàn bộ test users |
| `Recall` | Recall@k trên toàn bộ test users |
| `nDCG` | Normalized DCG@k |
| `MAP` | Mean Average Precision@k |
| `MRR` | Mean Reciprocal Rank@k |
| `Precision_u1` | Precision@k chỉ trên nhóm user có ≤ 1 interaction trong train |
| `Recall_u1` | Recall@k — nhóm user ≤ 1 train interaction |
| `nDCG_u1` | nDCG@k — nhóm user ≤ 1 train interaction |
| `MAP_u1` | MAP@k — nhóm user ≤ 1 train interaction |
| `MRR_u1` | MRR@k — nhóm user ≤ 1 train interaction |

Metric `_u1` trả về `NaN` nếu không có user nào trong nhóm (ví dụ: base split với k=20 kcore, tất cả user đều có nhiều hơn 1 train interaction).

Cutoffs mặc định: `[10, 20]`.
