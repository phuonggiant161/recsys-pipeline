"""
Quét data/processed/, tự sinh Elliot config (.yml) vào external/elliot/config_files/.

Cách dùng:
    python scripts/generate_elliot_configs.py
    python scripts/generate_elliot_configs.py --overwrite
    python scripts/generate_elliot_configs.py --filter hm_k20
"""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_OUT_DIR = PROJECT_ROOT / "external" / "elliot" / "config_files"

RESULTS_RECS = "../../../results/elliot/recs/"
RESULTS_PERF = "../../../results/elliot/performance/"
RESULTS_LOGS = "../../../results/elliot/logs/"


def make_config(folder_name: str) -> str:
    return f"""\
experiment:
  dataset: {folder_name}

  data_config:
    strategy: fixed
    train_path: ../../../data/processed/{folder_name}/train.tsv
    test_path: ../../../data/processed/{folder_name}/test.tsv

  top_k: 50

  evaluation:
    cutoffs: [10, 20]
    simple_metrics: [Precision, Recall, nDCG, MAP, MRR, Precision_u1, Recall_u1, nDCG_u1, MAP_u1, MRR_u1]

  path_output_rec_result: {RESULTS_RECS}
  path_output_rec_performance: {RESULTS_PERF}
  path_log_folder: {RESULTS_LOGS}

  gpu: -1

  models:
    ItemKNN:
      meta:
        save_recs: True
      neighbors: 40
      similarity: cosine
"""


def ensure_results_dirs() -> None:
    for subdir in ("recs", "performance", "logs"):
        (PROJECT_ROOT / "results" / "elliot" / subdir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Elliot config files from data/processed/")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing config files")
    parser.add_argument("--filter", default="", metavar="TEXT", help="Only generate configs for folders whose name contains TEXT")
    args = parser.parse_args()

    ensure_results_dirs()
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_DIR.exists():
        print(f"[ERROR] Directory not found: {PROCESSED_DIR}")
        return

    folders = sorted(p for p in PROCESSED_DIR.iterdir() if p.is_dir())

    if args.filter:
        folders = [p for p in folders if args.filter in p.name]

    generated = skipped_existing = skipped_missing = 0

    for folder in folders:
        if not (folder / "train.tsv").exists() or not (folder / "test.tsv").exists():
            print(f"[SKIP] {folder.name} - missing train.tsv or test.tsv")
            skipped_missing += 1
            continue

        out_path = CONFIG_OUT_DIR / f"{folder.name}_itemknn.yml"

        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} - already exists (use --overwrite to regenerate)")
            skipped_existing += 1
            continue

        out_path.write_text(make_config(folder.name), encoding="utf-8")
        print(f"[GEN]  {out_path.name}")
        generated += 1

    print(f"\nDone: {generated} generated, {skipped_existing} skipped (existing), {skipped_missing} skipped (missing tsv)")


if __name__ == "__main__":
    main()
