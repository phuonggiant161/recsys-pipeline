import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
np.float = float

_PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR    = _PROJECT_ROOT / "scripts"
_ARTIFACTS_ROOT = _PROJECT_ROOT / "results" / "selected_artifacts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from selected_artifact import read_artifact, print_artifact

# ---------------------------------------------------------------------------
# Metrics added on top of the checkpoint's original metric list
# ---------------------------------------------------------------------------

_pop_metrics = [
    'PopRecall_pop1',     'PopRecall_pop2_5',   'PopRecall_pop6_10',
    'PopRecall_pop11_20', 'PopRecall_pop20plus',
    'PopNDCG_pop1',       'PopNDCG_pop2_5',     'PopNDCG_pop6_10',
    'PopNDCG_pop11_20',   'PopNDCG_pop20plus',
]
_user_metrics = [
    'UserRecall_u1',     'UserRecall_u2_5',   'UserRecall_u6_10',
    'UserRecall_u11_20', 'UserRecall_u20plus',
    'UserNDCG_u1',       'UserNDCG_u2_5',     'UserNDCG_u6_10',
    'UserNDCG_u11_20',   'UserNDCG_u20plus',
]
_beyond_metrics = ['ItemCoverage', 'AveragePopularity', 'GiniIndex', 'TailPercentage']


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_beyond_accuracy_result(dataset_name, model_name, test_result):
    """Save beyond-accuracy metrics per cutoff to *_beyond.tsv using canonical column names.

    RecBole native → canonical:
      itemcoverage      → ItemCoverage
      averagepopularity → AveragePopularity
      giniindex         → Gini
      tailpercentage    → TailPercentage
    """
    _beyond_map = {
        'itemcoverage':      'ItemCoverage',
        'averagepopularity': 'AveragePopularity',
        'giniindex':         'Gini',
        'tailpercentage':    'TailPercentage',
    }
    _beyond_cols = ['ItemCoverage', 'AveragePopularity', 'Gini', 'TailPercentage']

    by_cutoff: dict = {}
    for key, val in test_result.items():
        if '@' not in str(key):
            continue
        metric_raw, cutoff_str = str(key).rsplit('@', 1)
        col = _beyond_map.get(metric_raw.strip().lower())
        if col is None:
            continue
        try:
            cutoff = int(cutoff_str.strip())
        except ValueError:
            continue
        by_cutoff.setdefault(cutoff, {})[col] = float(val)

    if not by_cutoff:
        return

    out_dir  = _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_beyond.tsv"

    rows = []
    for cutoff in sorted(by_cutoff):
        row = {'model': model_name, 'cutoff': cutoff}
        for col in _beyond_cols:
            row[col] = by_cutoff[cutoff].get(col, float('nan'))
        rows.append(row)

    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'cutoff'] + _beyond_cols, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    rel = out_file.relative_to(_PROJECT_ROOT)
    print(f"\n[evaluation] Beyond-accuracy metrics saved:\n  {rel}")


def _save_evaluation_result(dataset_name, model_name, ckpt_path, test_result):
    out_dir  = _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_evaluation.tsv"

    row = {"dataset": dataset_name, "model": model_name, "checkpoint": ckpt_path,
           **test_result}

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    rel = out_file.relative_to(_PROJECT_ROOT)
    print(f"\n[evaluation] Saved:\n  {rel}")


# ---------------------------------------------------------------------------
# Core evaluation for one checkpoint
# ---------------------------------------------------------------------------

def _run_one(
    checkpoint_path: str,
    overwrite: bool = False,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> str:
    # Fast skip when caller already knows dataset/model (avoids loading the checkpoint)
    if dataset_name and model_name:
        out_file = (
            _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
            / f"{model_name}_evaluation.tsv"
        )
        if out_file.exists() and not overwrite:
            print(f"  [SKIP] {out_file.relative_to(_PROJECT_ROOT)}  (use --overwrite)")
            return "skip"

    config, model, _, train_data, _, test_data = \
        load_data_and_model(model_file=checkpoint_path)

    dataset_name = config["dataset"]
    model_name   = config["model"]

    # Fallback skip check for the -f path where dataset/model were unknown before load
    out_file = (
        _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
        / f"{model_name}_evaluation.tsv"
    )
    if out_file.exists() and not overwrite:
        print(f"  [SKIP] {out_file.relative_to(_PROJECT_ROOT)}  (use --overwrite)")
        return "skip"

    # tail_ratio=0.1 → bottom 10% least-popular items (fraction branch in TailPercentage)
    config['tail_ratio'] = 0.1
    _additional = _pop_metrics + _user_metrics + _beyond_metrics
    config['metrics'] = list(dict.fromkeys(list(config['metrics']) + _additional))

    trainer = Trainer(config, model)
    trainer.eval_collector.data_collect(train_data)

    result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )
    print(result)

    _save_evaluation_result(dataset_name, model_name, checkpoint_path, result)
    _save_beyond_accuracy_result(dataset_name, model_name, result)
    return "ok"


# ---------------------------------------------------------------------------
# Dataset/model discovery for --all
# ---------------------------------------------------------------------------

def _discover_pairs(filter_str: str | None, model_filter: str | None) -> list[tuple[str, str]]:
    """Return RecBole-only (dataset, model) pairs from results/selected_artifacts/."""
    if not _ARTIFACTS_ROOT.exists():
        raise SystemExit(f"ERROR: {_ARTIFACTS_ROOT} does not exist — run recbole_run.py first.")
    pairs = []
    for ds_dir in sorted(_ARTIFACTS_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        if filter_str and filter_str.lower() not in dataset.lower():
            continue
        for artifact_file in sorted(ds_dir.glob("*.json")):
            try:
                meta = json.loads(artifact_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if meta.get("framework") != "recbole":
                continue
            model = artifact_file.stem
            if model_filter and model.lower() != model_filter.lower():
                continue
            pairs.append((dataset, model))
    if not pairs:
        raise SystemExit("No artifact pairs found matching the given filters.")
    return pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run RecBole post-hoc evaluation (group + beyond-accuracy metrics)."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-f", "--filetrained",
        metavar="CHECKPOINT",
        help="Path to a trained model checkpoint (.pth). Ignores --dataset/--model.",
    )
    source.add_argument(
        "--dataset",
        metavar="DATASET",
        help="Dataset name for metadata auto-lookup (requires --model).",
    )
    source.add_argument(
        "--all",
        action="store_true",
        help="Run on all (dataset, model) pairs found in results/selected_artifacts/.",
    )

    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Model name. Required with --dataset; optional filter with --all.",
    )
    parser.add_argument(
        "--filter",
        metavar="TEXT",
        help="(with --all) only datasets whose name contains TEXT.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if *_evaluation.tsv already exists.",
    )

    args = parser.parse_args()

    if args.filter and not args.all:
        parser.error("--filter can only be used with --all.")
    if args.filetrained and args.model:
        parser.error("--model cannot be used with -f/--filetrained.")

    # ── Single checkpoint via -f ──────────────────────────────────────────────
    if args.filetrained:
        _run_one(args.filetrained, overwrite=args.overwrite)
        return

    # ── Single dataset/model ──────────────────────────────────────────────────
    if args.dataset:
        if not args.model:
            parser.error("--dataset requires --model.")
        meta = read_artifact(args.dataset, args.model)
        print_artifact(meta)
        _run_one(
            meta["_resolved_path"],
            overwrite=args.overwrite,
            dataset_name=args.dataset,
            model_name=args.model,
        )
        return

    # ── --all ─────────────────────────────────────────────────────────────────
    pairs = _discover_pairs(args.filter, args.model)
    print(f"Found {len(pairs)} artifact(s) to evaluate:")
    for ds, m in pairs:
        print(f"  {ds}  /  {m}")
    print()

    success = skip = error = 0
    for dataset, model_name in pairs:
        print(f"\n{'='*60}")
        print(f"dataset={dataset}  model={model_name}")
        print("="*60)
        try:
            meta = read_artifact(dataset, model_name)
            print_artifact(meta)
            status = _run_one(
                meta["_resolved_path"],
                overwrite=args.overwrite,
                dataset_name=dataset,
                model_name=model_name,
            )
            if status == "skip":
                skip += 1
            else:
                success += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            error += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success} success, {skip} skip, {error} error")
    print("="*60)


if __name__ == "__main__":
    main()
