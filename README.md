# recsys-pipeline

Pipeline nghiên cứu ảnh hưởng của data sparsity đến chất lượng recommender system.

**Luồng chính:**
```
Raw data → k-core filtering → Train/Test split → Thinning → Elliot & RecBole training & evaluation
```

---

## Cấu trúc repo

```
recsys-pipeline/
├── preprocess.py                        # Pipeline chính (kcore → split → thinning)
├── src/
│   ├── config.py                  # Khai báo dataset (input path, cột)
│   ├── thinning.py                # 3 chiến lược thinning: random, head-item, tail-item
│   ├── dataset_folder.py          # Lưu train/valid/test + tự export .tsv cho Elliot
│   └── ...
├── scripts/
│   ├── generate_elliot_configs.py      # Tự sinh Elliot config từ data/processed/
│   ├── run_elliot.py                   # Chạy Elliot experiments từ repo root
│   ├── build_vsm_item_attributes.py    # Sinh item_attributes.tsv cho model VSM
│   ├── recbole_prepare_data.py         # Convert processed CSVs → RecBole atomic files (.inter)
│   └── recbole_run.py                  # Chạy RecBole experiments (BPR / ItemKNN)
├── configs/recbole/
│   ├── bpr.yml                         # Config model BPR (dùng chung cho mọi dataset)
│   └── itemknn.yml                     # Config model ItemKNN (dùng chung cho mọi dataset)
├── external/elliot/               # Elliot framework
├── data/
│   ├── raw/                       # Dữ liệu thô (không commit)
│   │   ├── hm.csv
│   │   ├── metadata_hm.csv        # Article attributes (article_id, prod_name, detail_desc, ...)
│   │   └── amazon.parquet
│   ├── processed/                 # Output của preprocess.py (không commit)
│   │   └── {folder}/
│   │       ├── train.tsv / valid.tsv / test.tsv   # Elliot format
│   │       └── item_attributes.tsv                # Cho VSM (sinh bởi build_vsm_item_attributes.py)
│   └── reports/                   # Sparsity summary CSV (commit)
└── results/elliot/
    └── {tên_dataset}/performance/ # Kết quả metrics của Elliot (commit)
```

---

## Chuẩn bị dữ liệu

File dữ liệu thô không được commit do kích thước lớn. Tải về tại:

https://drive.google.com/drive/folders/15D99fn2hSeKwfRGJWadF9hDNB4DaRqrb

Sau khi tải về, đặt vào:
```
data/raw/hm.csv
data/raw/amazon.parquet
```

---

## Cài đặt (1 lần duy nhất)

### 0. Windows — cho phép chạy script (nếu chưa làm)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 1. Cài thư viện

```bash
pip install -r requirements.txt
pip install -e external/elliot/ --no-deps
```

---

## Chạy pipeline

Sau khi cài đặt xong, mỗi lần chạy chỉ cần **3 lệnh**:

```bash
python preprocess.py                               # Bước 1: sinh data/processed/
python scripts/generate_elliot_configs.py    # Bước 2: sinh Elliot config YAML
python scripts/run_elliot.py                 # Bước 3: chạy Elliot
```

### Tùy chọn thêm

**Bước 1** — có thể sửa tham số trực tiếp trong `preprocess.py` (đầu file) hoặc truyền qua CLI:

Cách 1 — sửa constants trong file:
```python
DATASET_NAME = "hm"     # hoặc "amazon"
K_USER       = 20       # k-core: min interactions per user
K_ITEM       = 20       # k-core: min interactions per item
KEEP_FRACS   = ;
```

Cách 2 — truyền qua CLI (override constants):
```bash
# k_user = k_item = 20 (backward compat)
python preprocess.py --dataset hm --k 20

# k_user và k_item khác nhau
python preprocess.py --dataset hm --k-user 10 --k-item 30
python preprocess.py --dataset amazon --k-user 3 --k-item 10
```

**Bước 2** — các flag hữu ích:
```bash
python scripts/generate_elliot_configs.py --model ItemKNN            # chỉ gen ItemKNN
python scripts/generate_elliot_configs.py --model VSM                # chỉ gen VSM
python scripts/generate_elliot_configs.py --model ItemKNN VSM        # gen cả hai
python scripts/generate_elliot_configs.py --model ItemKNN --filter hm_random  # subset
python scripts/generate_elliot_configs.py --model ItemKNN --overwrite          # gen lại
```

**Bước 3** — các flag hữu ích:
```bash
python scripts/run_elliot.py --filter hm_random              # chỉ chạy subset
python scripts/run_elliot.py --config hm_random_keep0.05_itemknn     # 1 experiment
python scripts/run_elliot.py --config hm_random_keep0.05_vsm         # 1 experiment VSM
python scripts/run_elliot.py --filter hm_random --dry-run            # xem lệnh, không chạy
```

Kết quả ghi vào `results/elliot/{tên_dataset}/performance/`.

---

## VSM — Sinh item attributes

Cần chạy **một lần** sau `preprocess.py`, trước khi gen config VSM.
Script đọc các text columns từ metadata, vectorize bằng CountVectorizer, sinh file integer feature dùng chung cho toàn bộ experiments.

```bash
# H&M (item key: article_id, text: prod_name + detail_desc):
python scripts/build_vsm_item_attributes.py \
    --metadata-path data/raw/metadata_hm.csv \
    --global-output data/processed/hm_item_attributes.tsv \
    --dataset hm

# Amazon (item key: parent_asin, text: title + description):
python scripts/build_vsm_item_attributes.py \
    --metadata-path data/raw/metadata_amazon.csv \
    --global-output data/processed/amazon_item_attributes.tsv \
    --dataset amazon
```

Output: một file dùng chung cho tất cả experiments của dataset đó (format `item_id<TAB>feat_id_1<TAB>feat_id_2...`).

---

## RecBole — Benchmark bổ sung (source)

RecBole chạy song song với Elliot trên cùng dữ liệu đã split sẵn, kết quả lưu riêng tại `results/recbole/`.

Source RecBole nằm trong `external/recbole/`. Project wrapper nằm trong `scripts/recbole_*.py`.

Config model được định nghĩa **một lần duy nhất** tại `configs/recbole/bpr.yml` và `configs/recbole/itemknn.yml` — dùng chung cho mọi dataset, không cần sinh lại.

### Clone RecBole source (1 lần duy nhất)

```bash
git clone https://github.com/RUCAIBox/RecBole.git external/recbole
pip install -e external/recbole/
```

Nếu thiếu dependency:
```bash
pip install -r external/recbole/requirements.txt
```

> RecBole phụ thuộc PyTorch. BPR dùng `use_gpu: true`, ItemKNN dùng `use_gpu: false`.

### Chạy RecBole (BPR & ItemKNN)

**Bước 1 — Convert dữ liệu sang RecBole atomic files:**

```bash
# 1 dataset
python scripts/recbole_prepare_data.py --dataset hm_random_keep0.1 --overwrite

# Tất cả datasets
python scripts/recbole_prepare_data.py --all --overwrite
```

**Bước 2 — Chạy experiment:**

```bash
# 1 dataset, 1 model
python scripts/recbole_run.py --dataset hm_random_keep0.1 --model BPR --overwrite
python scripts/recbole_run.py --dataset hm_random_keep0.1 --model ItemKNN --overwrite

# 1 dataset, cả hai model
python scripts/recbole_run.py --dataset hm_random_keep0.1 --model all --overwrite

# Tất cả datasets
python scripts/recbole_run.py --all --model all --overwrite

# Tất cả datasets, lọc theo tên
python scripts/recbole_run.py --all --model ItemKNN --filter hm_random --overwrite
```

Kết quả ghi vào `results/recbole/<dataset>/performance/{BPR,ItemKNN}_recbole.tsv`.

**Ghi chú:**
- Config dùng `benchmark_filename: [train, valid, test]` — RecBole đọc split có sẵn, không tự split lại.
- Bỏ qua dataset đã có kết quả nếu không truyền `--overwrite`.
- Khi so sánh với Elliot: RecBole mask cả train+valid khi test, Elliot chỉ mask train — metrics có thể khác nhau một chút.

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

Metric `_u1` trả về `0.0` nếu không có user nào trong nhóm (ví dụ: base split với k=20 kcore, tất cả user đều có nhiều hơn 1 train interaction).

Cutoffs mặc định: `[10, 20, 50]`.
