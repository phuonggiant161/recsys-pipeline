"""
Run RecBole experiments via run_recbole() API and save metrics to TSV.

Usage:
    python scripts/recbole_run.py --dataset amazon_random_keep0.5 --model BPR --overwrite
    python scripts/recbole_run.py --dataset amazon_random_keep0.5 --model ItemKNN --overwrite
    python scripts/recbole_run.py --dataset amazon_random_keep0.5 --model all --overwrite
    python scripts/recbole_run.py --all --model BPR --overwrite
    python scripts/recbole_run.py --all --model ItemKNN --overwrite
    python scripts/recbole_run.py --all --model all --overwrite
    python scripts/recbole_run.py --all --model ItemKNN --filter amazon_random_keep0.5 --overwrite
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RECBOLE_SRC  = _PROJECT_ROOT / "external" / "recbole"
_SCRIPTS_DIR  = Path(__file__).resolve().parent

# external/recbole must be first in sys.path to shadow any installed RecBole
if str(_RECBOLE_SRC) not in sys.path:
    sys.path.insert(0, str(_RECBOLE_SRC))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from selected_artifact import write_artifact, print_artifact as _print_artifact

try:
    from recbole.quick_start import run_recbole
    import recbole as _recbole_pkg
    _RECBOLE_MODULE_PATH = Path(_recbole_pkg.__file__).parent
except ImportError as exc:
    raise SystemExit(
        f"ERROR: RecBole not importable ({exc}).\n"
        f"  Expected source at: {_RECBOLE_SRC}"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_TO_YAML: dict[str, Path] = {
    "BPR":     _PROJECT_ROOT / "configs" / "recbole" / "bpr.yml",
    "ItemKNN": _PROJECT_ROOT / "configs" / "recbole" / "itemknn.yml",
    "NeuMF":   _PROJECT_ROOT / "configs" / "recbole" / "neumf.yml",
    "LightGCN": _PROJECT_ROOT / "configs" / "recbole" / "lightgcn.yml",
}
_ALL_MODELS = list(_MODEL_TO_YAML.keys())

_DATA_ROOT       = _PROJECT_ROOT / "data" / "recbole"
_RESULTS_ROOT    = _PROJECT_ROOT / "results" / "recbole"
_CHECKPOINT_ROOT = _RESULTS_ROOT / "_checkpoints"

_REQUIRED_HEADER = {"user_id:token", "item_id:token", "timestamp:float"}

_BASE_METRICS = ["Precision", "Recall", "nDCG", "MAP", "MRR"]
_ALL_COLS     = ["model", "cutoff"] + _BASE_METRICS

_METRIC_MAP = {
    "precision": "Precision",
    "recall":    "Recall",
    "ndcg":      "nDCG",
    "map":       "MAP",
    "mrr":       "MRR",
}

_CUTOFFS           = [10, 20, 50]
_ALL_REQUIRED_KEYS = [f"{m}@{k}" for m in ["precision","recall","ndcg","map","mrr"] for k in _CUTOFFS]


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _check_inter_header(path: Path) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            cols = set(f.readline().strip().split("\t"))
        return _REQUIRED_HEADER <= cols
    except Exception:
        return False


def _discover_datasets(filter_str: str | None) -> list[str]:
    """Return sorted valid dataset names from data/recbole/."""
    if not _DATA_ROOT.exists():
        raise SystemExit(
            f"ERROR: {_DATA_ROOT} does not exist.\n"
            "  Run: python scripts/recbole_prepare_data.py --all --overwrite"
        )
    datasets = []
    for d in sorted(_DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if filter_str and filter_str not in name:
            continue
        missing     = [s for s in ["train","valid","test"] if not (d / f"{name}.{s}.inter").exists()]
        bad_header  = [s for s in ["train","valid","test"]
                       if not missing and not _check_inter_header(d / f"{name}.{s}.inter")]
        if missing:
            print(f"  [SKIP] dataset={name}  reason=missing inter files: {missing}")
        elif bad_header:
            print(f"  [SKIP] dataset={name}  reason=bad header in: {bad_header}")
        else:
            datasets.append(name)
    if not datasets:
        suffix = f" matching '{filter_str}'" if filter_str else ""
        raise SystemExit(f"No valid datasets found in {_DATA_ROOT}{suffix}")
    return datasets


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

def _parse_test_result(test_result: dict, model_name: str) -> list[dict]:
    """Parse RecBole test_result into per-cutoff rows. Raises on missing keys."""
    missing = [k for k in _ALL_REQUIRED_KEYS if k not in test_result]
    if missing:
        raise RuntimeError(
            f"Missing metric keys from test_result: {missing}\n"
            f"  Available keys: {sorted(test_result.keys())}"
        )

    by_cutoff: dict[int, dict] = {}
    for key, val in test_result.items():
        if "@" not in key:
            continue
        metric_raw, cutoff_str = key.rsplit("@", 1)
        col = _METRIC_MAP.get(metric_raw.strip().lower())
        if col is None:
            continue
        try:
            cutoff = int(cutoff_str.strip())
        except ValueError:
            continue
        by_cutoff.setdefault(cutoff, {})[col] = float(val)

    rows = []
    for cutoff in sorted(by_cutoff):
        row: dict = {"model": model_name, "cutoff": cutoff}
        for col in _BASE_METRICS:
            row[col] = by_cutoff[cutoff][col]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def _run_one(
    dataset: str,
    model_name: str,
    overwrite: bool,
    show_progress_bar: bool,
) -> str:
    out_dir  = _RESULTS_ROOT / dataset / "performance"
    out_file = out_dir / f"{model_name}_recbole.tsv"

    if out_file.exists() and not overwrite:
        print(f"  [SKIP] {out_file.relative_to(_PROJECT_ROOT)} (use --overwrite)")
        return "skip"

    config_path    = _MODEL_TO_YAML[model_name]
    checkpoint_dir = _CHECKPOINT_ROOT / dataset / model_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_path = str(_DATA_ROOT) + "/"

    print(f"  RecBole source : {_RECBOLE_MODULE_PATH}")
    print(f"  Config         : {config_path.relative_to(_PROJECT_ROOT)}")
    print(f"  data_path      : {data_path}")
    print(f"  checkpoint_dir : {checkpoint_dir.relative_to(_PROJECT_ROOT)}")
    print(f"  Output         : {out_file.relative_to(_PROJECT_ROOT)}")

    result = run_recbole(
        model=model_name,
        dataset=dataset,
        config_file_list=[str(config_path)],
        config_dict={
            "data_path":      data_path,
            "checkpoint_dir": str(checkpoint_dir),
            "show_progress":  show_progress_bar,
        },
        saved=True,
    )

    if result is None or "test_result" not in result:
        raise RuntimeError("run_recbole() returned None or missing 'test_result'")

    # Write selected-artifact metadata immediately after training
    checkpoint_path = result.get("checkpoint_path")
    if checkpoint_path:
        _, meta = write_artifact(
            framework="recbole",
            dataset=dataset,
            model=model_name,
            selected_artifact=checkpoint_path,
            artifact_type="checkpoint",
        )
        _print_artifact(meta)
    else:
        print("  [selected_artifact] WARN: checkpoint_path missing from result — metadata not written.")

    test_result = result["test_result"]
    rows = _parse_test_result(test_result, model_name)

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=_ALL_COLS)
    df.to_csv(out_file, sep="\t", index=False)

    print(f"\n  Results -> {out_file.relative_to(_PROJECT_ROOT)}")
    print(df.to_string(index=False))
    return "ok"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    valid_names = " | ".join(_ALL_MODELS)
    p = argparse.ArgumentParser(
        description=f"Run RecBole experiments ({valid_names}) and save metrics TSV"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset name (must exist in data/recbole/)")
    group.add_argument("--all",     action="store_true",
                       help="Run on all valid datasets in data/recbole/")
    p.add_argument("--model",    required=True,
                   help=f"Model to run: {valid_names} | all")
    p.add_argument("--filter",   help="(with --all) only datasets whose name contains this substring")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing TSV output (checked per dataset×model)")
    p.add_argument(
        "--noprogressbar",
        action="store_true",
        help="Turn off progress bar"
    )
    args = p.parse_args()

    # Resolve model list — case-insensitive lookup
    model_arg = args.model.strip().lower()
    _name_map = {k.lower(): k for k in _ALL_MODELS}
    if model_arg == "all":
        models = _ALL_MODELS
    elif model_arg in _name_map:
        models = [_name_map[model_arg]]
    else:
        raise SystemExit(
            f"ERROR: --model must be one of: {valid_names}, all. Got: {args.model!r}"
        )

    # Validate YAML files exist
    for m in models:
        cfg = _MODEL_TO_YAML[m]
        if not cfg.exists():
            raise SystemExit(f"ERROR: Config not found: {cfg}")

    # Resolve dataset list
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = _discover_datasets(args.filter)

    success = skip = error = 0
    errors: list[tuple[str, str, str]] = []
    experiments = [(ds, m) for ds in datasets for m in models]

    for dataset, model_name in experiments:
        print(f"\n{'='*60}")
        print(f"dataset={dataset}  model={model_name}")
        print("="*60)
        try:
            status = _run_one(
                dataset,
                model_name,
                args.overwrite,
                show_progress_bar=not args.noprogressbar,
            )
            if status == "skip":
                skip += 1
            else:
                success += 1
        except Exception as e:
            msg = str(e)
            print(f"  [ERROR] {msg}")
            errors.append((dataset, model_name, msg))
            error += 1

    if len(experiments) > 1:
        print(f"\n{'='*60}")
        print(f"Summary: {success} success, {skip} skip, {error} error")
        if errors:
            for ds, m, msg in errors:
                print(f"  FAIL  dataset={ds}  model={m}  config={_MODEL_TO_YAML[m].name}")
                print(f"        {msg.splitlines()[0][:120]}")
        print("="*60)


if __name__ == "__main__":
    main()
