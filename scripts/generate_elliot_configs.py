"""
Quét data/processed/, tự sinh Elliot config (.yml) vào external/elliot/config_files/.

Path resolution trong Elliot (quan trọng):
- train_path, test_path, attribute_file
    → resolve qua _safe_set_path() từ thư mục config_files/
    → cần prefix ../../../  (3 cấp lên đến project root)
- path_output_rec_result, path_output_rec_performance, path_log_folder
    → resolve qua os.path.abspath() từ CWD = external/elliot/
    → cần prefix ../../    (2 cấp lên đến project root)

Cách dùng:
    python scripts/generate_elliot_configs.py --model ItemKNN
    python scripts/generate_elliot_configs.py --model VSM
    python scripts/generate_elliot_configs.py --model ItemKNN VSM
    python scripts/generate_elliot_configs.py --model VSM --filter hm_random --overwrite
"""
import argparse
from pathlib import Path
from typing import Callable, NamedTuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIG_OUT_DIR = PROJECT_ROOT / "external" / "elliot" / "config_files"

# Paths resolved from config_files/ (via _safe_set_path): 3 levels up to project root
_DATA = "../../../data/processed"

# Paths resolved from CWD = external/elliot/ (via os.path.abspath): 2 levels up
_RECS_FROM_CWD = "../../results/elliot"

# Shared item attribute files (one per dataset, not per experiment folder).
# Key = folder name prefix, value = path relative to config_files/ (3 levels up).
_SHARED_ATTR: dict[str, str] = {
    "hm":     "../../../data/processed/hm_item_attributes.tsv",
    "amazon": "../../../data/processed/amazon_item_attributes.tsv",
}


def _attr_path_for(folder_name: str) -> str:
    """Return attribute_file path for a given experiment folder."""
    for prefix, path in _SHARED_ATTR.items():
        if folder_name == prefix or folder_name.startswith(prefix + "_"):
            return path
    return f"{_DATA}/{folder_name}/item_attributes.tsv"


def _attr_local_path_for(folder_name: str) -> Path:
    """Return the absolute Path of the attribute file for preflight checks."""
    for prefix in _SHARED_ATTR:
        if folder_name == prefix or folder_name.startswith(prefix + "_"):
            return PROJECT_ROOT / "data" / "processed" / f"{prefix}_item_attributes.tsv"
    return PROCESSED_DIR / folder_name / "item_attributes.tsv"


# ── Model definition ──────────────────────────────────────────────────────────

class _Model(NamedTuple):
    suffix: str
    # Returns indented YAML block for under `models:` (4-space indent)
    models_yaml: Callable[[str], str]
    # Returns extra YAML to append inside data_config: after test_path ("" = nothing)
    data_config_extra: Callable[[str], str]
    # Returns None if folder is ready; error message string if not
    preflight: Callable[[str], Optional[str]]


# ── ItemKNN ───────────────────────────────────────────────────────────────────

def _itemknn_models_yaml(_: str) -> str:
    return """\
    ItemKNN:
      meta:
        save_recs: True
      neighbors: 40
      similarity: cosine"""


ITEMKNN = _Model(
    suffix="itemknn",
    models_yaml=_itemknn_models_yaml,
    data_config_extra=lambda _: "",
    preflight=lambda _: None,
)


# ── VSM ───────────────────────────────────────────────────────────────────────

def _vsm_models_yaml(_: str) -> str:
    return """\
    VSM:
      meta:
        save_recs: True
      similarity: cosine"""


def _vsm_data_config_extra(folder_name: str) -> str:
    return (
        "\n    side_information:\n"
        "    - dataloader: ItemAttributes\n"
        f"      attribute_file: {_attr_path_for(folder_name)}"
    )


def _vsm_preflight(folder_name: str) -> Optional[str]:
    attr_file = _attr_local_path_for(folder_name)
    if not attr_file.exists():
        # Build the hint based on whether this is a known shared-attr dataset
        for prefix in _SHARED_ATTR:
            if folder_name == prefix or folder_name.startswith(prefix + "_"):
                return (
                    f"Missing data/processed/{prefix}_item_attributes.tsv — run first:\n"
                    f"  python scripts/build_vsm_item_attributes.py "
                    f"--metadata-path data/raw/metadata_{prefix}.csv "
                    f"--global-output data/processed/{prefix}_item_attributes.tsv "
                    f"--dataset {prefix}"
                )
        return (
            f"Missing item_attributes.tsv for {folder_name} — run first:\n"
            f"  python scripts/build_vsm_item_attributes.py "
            f"--metadata-path <metadata.csv> --dataset-dir data/processed/{folder_name}"
        )
    return None


VSM = _Model(
    suffix="vsm",
    models_yaml=_vsm_models_yaml,
    data_config_extra=_vsm_data_config_extra,
    preflight=_vsm_preflight,
)


# ── Registry ──────────────────────────────────────────────────────────────────
# Keys are lowercase; --model flag is matched case-insensitively.

MODELS: dict[str, _Model] = {
    "itemknn": ITEMKNN,
    "vsm":     VSM,
}


# ── Config builder ────────────────────────────────────────────────────────────

def _build_config(folder_name: str, model: _Model) -> str:
    data_extra = model.data_config_extra(folder_name)
    models_block = model.models_yaml(folder_name)
    return f"""\
experiment:
  dataset: {folder_name}

  data_config:
    strategy: fixed
    train_path: {_DATA}/{folder_name}/train.tsv
    validation_path: {_DATA}/{folder_name}/valid.tsv
    test_path: {_DATA}/{folder_name}/test.tsv{data_extra}

  top_k: 50

  evaluation:
    cutoffs: [10, 20, 50]
    simple_metrics: [Precision, Recall, nDCG, MAP, MRR, Precision_u1, Recall_u1, nDCG_u1, MAP_u1, MRR_u1]

  path_output_rec_result: {_RECS_FROM_CWD}/{folder_name}/recs/
  path_output_rec_performance: {_RECS_FROM_CWD}/{folder_name}/performance/
  path_log_folder: {_RECS_FROM_CWD}/{folder_name}/logs/

  gpu: -1

  models:
{models_block}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_results_dirs() -> None:
    (PROJECT_ROOT / "results" / "elliot").mkdir(parents=True, exist_ok=True)


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_train_configs(folders, models: list[_Model], overwrite: bool) -> None:
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = skipped_existing = skipped_missing = skipped_preflight = 0

    for folder in folders:
        missing = [f for f in ("train.tsv", "valid.tsv", "test.tsv") if not (folder / f).exists()]
        if missing:
            print(f"[SKIP] {folder.name} — missing {', '.join(missing)}")
            skipped_missing += 1
            continue

        for model in models:
            out_path = CONFIG_OUT_DIR / f"{folder.name}_{model.suffix}.yml"

            if out_path.exists() and not overwrite:
                print(f"[SKIP] {out_path.name} — already exists (use --overwrite)")
                skipped_existing += 1
                continue

            err = model.preflight(folder.name)
            if err:
                print(f"[SKIP] {out_path.name} — {err}")
                skipped_preflight += 1
                continue

            out_path.write_text(_build_config(folder.name, model), encoding="utf-8")
            print(f"[GEN]  {out_path.name}")
            generated += 1

    print(
        f"\nDone: {generated} generated, {skipped_existing} skipped (existing), "
        f"{skipped_missing} skipped (missing tsv), {skipped_preflight} skipped (preflight)"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    valid_names = ", ".join(MODELS)
    parser = argparse.ArgumentParser(description="Generate Elliot config files from data/processed/")
    parser.add_argument(
        "--model", nargs="+", metavar="NAME", default=["ItemKNN"],
        help=f"Model(s) to generate: {valid_names} (default: ItemKNN). Example: --model ItemKNN VSM",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing config files")
    parser.add_argument(
        "--filter", default="", metavar="TEXT",
        help="Only generate configs for folders whose name contains TEXT",
    )
    args = parser.parse_args()

    ensure_results_dirs()

    if not PROCESSED_DIR.exists():
        print(f"[ERROR] Directory not found: {PROCESSED_DIR}")
        return

    folders = sorted(p for p in PROCESSED_DIR.iterdir() if p.is_dir())
    if args.filter:
        folders = [p for p in folders if args.filter in p.name]

    selected: list[_Model] = []
    for name in args.model:
        key = name.lower()
        if key not in MODELS:
            print(f"[ERROR] Unknown model '{name}'. Available: {valid_names}")
            return
        selected.append(MODELS[key])

    model_names = ", ".join(m.suffix.upper() for m in selected)
    print(f"Mode: train ({model_names}) -> {CONFIG_OUT_DIR.relative_to(PROJECT_ROOT)}")
    generate_train_configs(folders, models=selected, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
