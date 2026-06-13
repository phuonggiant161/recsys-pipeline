"""
Quét data/processed/, tự sinh Elliot config (.yml) vào external/elliot/config_files/.

Path resolution trong Elliot (quan trọng):
- train_path, test_path
    → resolve qua _safe_set_path() từ thư mục config_files/ (hoặc config_files_eval/)
    → cần prefix ../../../  (3 cấp lên đến project root)
- path_output_rec_result, path_output_rec_performance, path_log_folder, RecommendationFolder.folder
    → resolve qua os.path.abspath() hoặc os.listdir() từ CWD = external/elliot/
    → cần prefix ../../    (2 cấp lên đến project root)

Cách dùng:
    python scripts/generate_elliot_configs.py               # train configs (ItemKNN)
    python scripts/generate_elliot_configs.py --overwrite
    python scripts/generate_elliot_configs.py --filter hm_k20

    python scripts/generate_elliot_configs.py --eval-only   # eval-only configs
    python scripts/generate_elliot_configs.py --eval-only --filter hm_k20
    python scripts/generate_elliot_configs.py --eval-only --overwrite
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_OUT_DIR = PROJECT_ROOT / "external" / "elliot" / "config_files"
EVAL_CONFIG_OUT_DIR = PROJECT_ROOT / "external" / "elliot" / "config_files_eval"

# Paths resolved from config_files/ dir (via _safe_set_path): 3 levels up
_DATA = "../../../data/processed"

# Paths resolved from CWD = external/elliot/ (via os.path.abspath or os.listdir): 2 levels up
# Used for: path_output_* fields AND RecommendationFolder.folder (fill_model uses raw path + os.listdir)
_RECS_FROM_CWD = "../../results/elliot"


def make_config(folder_name: str) -> str:
    return f"""\
experiment:
  dataset: {folder_name}

  data_config:
    strategy: fixed
    train_path: {_DATA}/{folder_name}/train.tsv
    test_path: {_DATA}/{folder_name}/test.tsv

  top_k: 50

  evaluation:
    cutoffs: [10, 20, 50]
    simple_metrics: [Precision, Recall, nDCG, MAP, MRR, Precision_u1, Recall_u1, nDCG_u1, MAP_u1, MRR_u1]

  path_output_rec_result: {_RECS_FROM_CWD}/{folder_name}/recs/
  path_output_rec_performance: {_RECS_FROM_CWD}/{folder_name}/performance/
  path_log_folder: {_RECS_FROM_CWD}/{folder_name}/logs/

  gpu: -1

  models:
    ItemKNN:
      meta:
        save_recs: True
      neighbors: 40
      similarity: cosine
"""


def make_eval_config(folder_name: str) -> str:
    return f"""\
experiment:
  dataset: {folder_name}

  data_config:
    strategy: fixed
    train_path: {_DATA}/{folder_name}/train.tsv
    test_path: {_DATA}/{folder_name}/test.tsv

  top_k: 50

  evaluation:
    cutoffs: [10, 20, 50]
    simple_metrics: [Precision, Recall, nDCG, MAP, MRR, Precision_u1, Recall_u1, nDCG_u1, MAP_u1, MRR_u1]

  path_output_rec_result: {_RECS_FROM_CWD}/{folder_name}/recs/
  path_output_rec_performance: {_RECS_FROM_CWD}/{folder_name}/performance/
  path_log_folder: {_RECS_FROM_CWD}/{folder_name}/logs/

  gpu: -1

  models:
    RecommendationFolder:
      folder: {_RECS_FROM_CWD}/{folder_name}/recs/
"""


def ensure_results_dirs() -> None:
    (PROJECT_ROOT / "results" / "elliot").mkdir(parents=True, exist_ok=True)


def _recs_dir(folder_name: str) -> Path:
    return PROJECT_ROOT / "results" / "elliot" / folder_name / "recs"


def generate_train_configs(folders, overwrite: bool) -> None:
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = skipped_existing = skipped_missing = 0

    for folder in folders:
        if not (folder / "train.tsv").exists() or not (folder / "test.tsv").exists():
            print(f"[SKIP] {folder.name} - missing train.tsv or test.tsv")
            skipped_missing += 1
            continue

        out_path = CONFIG_OUT_DIR / f"{folder.name}_itemknn.yml"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path.name} - already exists (use --overwrite to regenerate)")
            skipped_existing += 1
            continue

        out_path.write_text(make_config(folder.name), encoding="utf-8")
        print(f"[GEN]  {out_path.name}")
        generated += 1

    print(f"\nDone: {generated} generated, {skipped_existing} skipped (existing), {skipped_missing} skipped (missing tsv)")


def generate_eval_configs(folders, overwrite: bool) -> None:
    EVAL_CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = skipped_existing = skipped_no_recs = skipped_missing = 0

    for folder in folders:
        if not (folder / "train.tsv").exists() or not (folder / "test.tsv").exists():
            print(f"[SKIP] {folder.name} - missing train.tsv or test.tsv")
            skipped_missing += 1
            continue

        recs_dir = _recs_dir(folder.name)
        rec_files = list(recs_dir.glob("*.tsv")) if recs_dir.exists() else []

        if not rec_files:
            print(f"[SKIP] {folder.name} - no rec files in results/elliot/{folder.name}/recs/")
            skipped_no_recs += 1
            continue

        out_path = EVAL_CONFIG_OUT_DIR / f"{folder.name}_eval.yml"

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path.name} - already exists (use --overwrite to regenerate)")
            skipped_existing += 1
            continue

        out_path.write_text(make_eval_config(folder.name), encoding="utf-8")
        print(f"[GEN]  {out_path.name}  ({len(rec_files)} rec file(s) found)")
        for f in rec_files:
            print(f"         rec: {f.name}")
        generated += 1

    print(f"\nDone: {generated} generated, {skipped_existing} skipped (existing), "
          f"{skipped_no_recs} skipped (no recs), {skipped_missing} skipped (missing tsv)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Elliot config files from data/processed/")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing config files")
    parser.add_argument("--filter", default="", metavar="TEXT",
                        help="Only generate configs for folders whose name contains TEXT")
    parser.add_argument("--eval-only", action="store_true",
                        help="Generate eval-only configs (RecommendationFolder) instead of train configs")
    args = parser.parse_args()

    ensure_results_dirs()

    if not PROCESSED_DIR.exists():
        print(f"[ERROR] Directory not found: {PROCESSED_DIR}")
        return

    folders = sorted(p for p in PROCESSED_DIR.iterdir() if p.is_dir())
    if args.filter:
        folders = [p for p in folders if args.filter in p.name]

    if args.eval_only:
        print(f"Mode: eval-only (RecommendationFolder) -> {EVAL_CONFIG_OUT_DIR.relative_to(PROJECT_ROOT)}")
        generate_eval_configs(folders, overwrite=args.overwrite)
    else:
        print(f"Mode: train (ItemKNN) -> {CONFIG_OUT_DIR.relative_to(PROJECT_ROOT)}")
        generate_train_configs(folders, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
