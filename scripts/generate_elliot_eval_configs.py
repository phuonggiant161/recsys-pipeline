"""
Generate Elliot post-hoc evaluation configs for FunkSVD and VSM.

Base configuration is defined inline in _BASE_CONFIG — no external template file.
Output: external/elliot/config_files/evaluation/<dataset>_<model>.yml

Path resolution in Elliot:
  data_config paths (train/valid/test)  → config-file-relative  (4 ../ from evaluation/)
  path_output_rec_performance           → CWD-relative           (project root when invoked)
  path_log_folder                       → config-file-relative  (4 ../ from evaluation/)

Usage:
    python scripts/generate_elliot_eval_configs.py
    python scripts/generate_elliot_eval_configs.py --filter hm_random
"""
import argparse
import copy
from pathlib import Path

import yaml

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_EVAL_OUT_DIR  = _PROJECT_ROOT / "external" / "elliot" / "config_files" / "evaluation"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_EVAL_MODELS = ["FunkSVD", "VSM"]

# evaluation/ is one level deeper than config_files/ → 4 ../ to reach project root
_DATA_PREFIX = "../../../../data/processed"
_LOGS_PREFIX = "../../../../results/elliot"

# ── Base config ───────────────────────────────────────────────────────────────
# Single source of truth for metric list and evaluation settings.
# _make_config() deepcopies this and substitutes dataset/model fields.
# No external template file is read or required.

_BASE_CONFIG: dict = {
    "experiment": {
        "dataset": None,
        "data_config": {
            "strategy": "fixed",
            "train_path":      None,
            "validation_path": None,
            "test_path":       None,
        },
        "top_k": 50,
        "evaluation": {
            "cutoffs": [10, 20, 50],
            "simple_metrics": [
                "SEntropy",
                "Gini",
                "ItemCoverage",
                "ARP",
                "APLT",
                "Recall",
                "nDCG",
                "Recall_pop1",
                "Recall_pop2_5",
                "Recall_pop6_10",
                "Recall_pop11_20",
                "Recall_pop20plus",
                "nDCG_pop1",
                "nDCG_pop2_5",
                "nDCG_pop6_10",
                "nDCG_pop11_20",
                "nDCG_pop20plus",
                "Recall_u1",
                "Recall_u2_5",
                "Recall_u6_10",
                "Recall_u11_20",
                "Recall_u20plus",
                "nDCG_u1",
                "nDCG_u2_5",
                "nDCG_u6_10",
                "nDCG_u11_20",
                "nDCG_u20plus",
            ],
        },
        "path_output_rec_performance": None,
        "path_log_folder":             None,
        "gpu": 0,
        "models": {
            "ProxyRecommender": {
                "source_model": None,
                "path": "selected_artifact",
            }
        },
    }
}


# ── Dataset discovery ─────────────────────────────────────────────────────────

def discover_datasets(filter_str: str = "") -> list[str]:
    """Return sorted dataset names from data/processed/ that have all 3 split TSVs."""
    if not _PROCESSED_DIR.exists():
        raise SystemExit(f"ERROR: {_PROCESSED_DIR} does not exist")
    result = []
    for d in sorted(_PROCESSED_DIR.iterdir()):
        if not d.is_dir():
            continue
        if filter_str and filter_str not in d.name:
            continue
        missing = [s for s in ("train", "valid", "test") if not (d / f"{s}.tsv").exists()]
        if missing:
            print(f"  [SKIP] {d.name} — missing: {', '.join(m + '.tsv' for m in missing)}")
        else:
            result.append(d.name)
    return result


# ── Config builder ────────────────────────────────────────────────────────────

def _make_config(dataset: str, model_name: str) -> dict:
    """Return a new config dict for the given dataset and model.

    Deepcopies _BASE_CONFIG so the module-level constant is never mutated.
    """
    cfg = copy.deepcopy(_BASE_CONFIG)
    exp = cfg["experiment"]

    exp["dataset"] = dataset

    dc = exp["data_config"]
    dc["train_path"]      = f"{_DATA_PREFIX}/{dataset}/train.tsv"
    dc["validation_path"] = f"{_DATA_PREFIX}/{dataset}/valid.tsv"
    dc["test_path"]       = f"{_DATA_PREFIX}/{dataset}/test.tsv"

    exp["path_output_rec_performance"] = f"results/elliot/{dataset}/performance/"
    exp["path_log_folder"]             = f"{_LOGS_PREFIX}/{dataset}/logs/"

    exp["models"]["ProxyRecommender"]["source_model"] = model_name

    return cfg


# ── Generator ─────────────────────────────────────────────────────────────────

def generate(filter_str: str = "") -> tuple[int, int]:
    """Generate evaluation YAML configs. Returns (generated, skipped)."""
    datasets = discover_datasets(filter_str)
    if not datasets:
        print("  No valid datasets found.")
        return 0, 0

    print(f"  Datasets: {len(datasets)}")

    _EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = 0
    for dataset in datasets:
        for model_name in _EVAL_MODELS:
            cfg      = _make_config(dataset, model_name)
            out_file = _EVAL_OUT_DIR / f"{dataset}_{model_name.lower()}.yml"
            with open(out_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"  [GEN]  {out_file.name}")
            generated += 1

    return generated, 0


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Elliot evaluation configs for FunkSVD and VSM"
    )
    parser.add_argument(
        "--filter", default="", metavar="TEXT",
        help="Only generate for datasets whose name contains TEXT"
    )
    args = parser.parse_args()

    print(f"Output : {_EVAL_OUT_DIR.relative_to(_PROJECT_ROOT)}")
    print(f"Models : {', '.join(_EVAL_MODELS)}")
    print()

    generated, _ = generate(args.filter)
    print(f"\nDone: {generated} YAML file(s) generated.")


if __name__ == "__main__":
    main()
