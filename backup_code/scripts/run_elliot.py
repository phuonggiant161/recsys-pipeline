"""
Run Elliot experiments from the repo root.

Usage:
    python scripts/run_elliot.py
    python scripts/run_elliot.py --filter hm_random_keep0.05
    python scripts/run_elliot.py --config hm_random_keep0.05_itemknn hm_random_keep0.05_vsm
    python scripts/run_elliot.py --filter hm_random_keep0.05 --dry-run
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ELLIOT_DIR = PROJECT_ROOT / "external" / "elliot"
CONFIG_DIR = ELLIOT_DIR / "config_files"


def get_python() -> str:
    """Use venv_elliot if it exists, otherwise fall back to current Python."""
    for candidate in [
        ELLIOT_DIR / "venv_elliot" / "Scripts" / "python.exe",  # Windows
        ELLIOT_DIR / "venv_elliot" / "bin" / "python",          # Linux/Mac
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def collect_configs(filter_text: str, specific: list[str]) -> list[str]:
    """Return list of config stems to run, in sorted order."""
    if specific:
        return sorted(specific)

    if not CONFIG_DIR.exists():
        return []

    stems = sorted(p.stem for p in CONFIG_DIR.glob("*.yml"))
    if filter_text:
        stems = [s for s in stems if filter_text in s]
    return stems


# ── Performance file normalizer ───────────────────────────────────────────────

def _normalize_model_name(raw: str) -> str:
    """Extract short model name from Elliot's full model string.

    'ItemKNN_nn=40_sim=cosine_...' -> 'ItemKNN'
    'VSM_sim=cosine_up=tfidf_...' -> 'VSM'
    'MostPop'                      -> 'MostPop'
    """
    base = raw.split("_")[0] if "_" in raw else raw
    return re.sub(r"[^a-zA-Z0-9]", "", base)


def _get_perf_dir(stem: str) -> Path | None:
    """Extract performance directory for a config stem by reading its YAML."""
    config_path = CONFIG_DIR / f"{stem}.yml"
    if not config_path.exists():
        return None
    content = config_path.read_text(encoding="utf-8")
    m = re.search(r"path_output_rec_performance:\s*(.+)", content)
    if not m:
        return None
    # path is relative to ELLIOT_DIR (e.g. ../../results/elliot/{dataset}/performance/)
    return (ELLIOT_DIR / m.group(1).strip()).resolve()


def _normalize_performance_files(perf_dir: Path) -> None:
    """Rename Elliot timestamp-named TSV/JSON output to <Model>_cutoff_<N>.[tsv|json].

    rec_cutoff_10_relthreshold_0_2026_06_20_01_24_11.tsv
        -> ItemKNN_cutoff_10.tsv  (or VSM_cutoff_10.tsv, etc.)

    bestmodelparams_cutoff_10_relthreshold_0_2026_06_20.json
        -> bestmodelparams_ItemKNN_cutoff_10.json
    """
    if not perf_dir.exists():
        return

    try:
        import pandas as pd
    except ImportError:
        print("  [WARN] pandas not available — skipping performance rename")
        return

    renamed = 0

    # ── TSV files: rec_cutoff_<N>_relthreshold_... ────────────────────────────
    # Sort by mtime ascending so the newest file for each model-cutoff wins.
    tsv_files = sorted(
        perf_dir.glob("rec_cutoff_*.tsv"),
        key=lambda p: p.stat().st_mtime,
    )
    for tsv in tsv_files:
        m = re.search(r"rec_cutoff_(\d+)_", tsv.name)
        if not m:
            continue
        cutoff = m.group(1)

        try:
            df = pd.read_csv(tsv, sep="\t")
        except Exception as e:
            print(f"  [WARN] Could not read {tsv.name}: {e}")
            continue

        if "model" not in df.columns or df.empty:
            continue

        for _, row in df.iterrows():
            model_name = _normalize_model_name(str(row["model"]))
            out_path = perf_dir / f"{model_name}_cutoff_{cutoff}.tsv"
            pd.DataFrame([row]).to_csv(out_path, sep="\t", index=False)

        tsv.unlink()
        renamed += 1

    # ── JSON files: bestmodelparams_cutoff_<N>_relthreshold_... ──────────────
    json_files = sorted(
        perf_dir.glob("bestmodelparams_cutoff_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for jf in json_files:
        m = re.search(r"cutoff_(\d+)_", jf.name)
        if not m:
            continue
        cutoff = m.group(1)

        try:
            with open(jf, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [WARN] Could not read {jf.name}: {e}")
            continue

        model_name = None
        for entry in data:
            if isinstance(entry, dict) and "recommender" in entry:
                model_name = _normalize_model_name(entry["recommender"])
                break

        if model_name is None:
            continue

        out_path = perf_dir / f"bestmodelparams_{model_name}_cutoff_{cutoff}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        jf.unlink()
        renamed += 1

    if renamed:
        print(f"  [PERF] Normalized {renamed} performance file(s) in {perf_dir.parent.name}/performance/")


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_one(stem: str, dry_run: bool) -> bool:
    """Run a single experiment. Returns True on success."""
    cmd = [get_python(), "start_experiments.py", "--config", stem, "--config-dir", "config_files"]

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}[TRAIN] {stem}")
    print(f"  config : config_files/{stem}.yml")
    print(f"  cmd    : {' '.join(cmd)}")
    print(f"  cwd    : {ELLIOT_DIR}")

    if dry_run:
        return True

    result = subprocess.run(cmd, cwd=ELLIOT_DIR)
    if result.returncode != 0:
        print(f"[FAILED] {stem} (exit code {result.returncode})")
        return False

    perf_dir = _get_perf_dir(stem)
    if perf_dir:
        _normalize_performance_files(perf_dir)

    print(f"[OK]    {stem}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Elliot experiments from repo root")
    parser.add_argument(
        "--config", nargs="+", metavar="STEM", default=[],
        help="Config stem names to run (no .yml extension)",
    )
    parser.add_argument(
        "--filter", default="", metavar="TEXT",
        help="Run all configs whose name contains TEXT",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without running",
    )
    args = parser.parse_args()

    stems = collect_configs(filter_text=args.filter, specific=args.config)

    if not stems:
        print("No configs found.")
        print("Generate configs first: python scripts/generate_elliot_configs.py --model ItemKNN")
        sys.exit(1)

    print(f"Mode: TRAIN | Experiments to run ({len(stems)}):")
    for s in stems:
        print(f"  {s}")

    failed = []
    for stem in stems:
        ok = run_one(stem, dry_run=args.dry_run)
        if not ok:
            failed.append(stem)

    print(f"\n{'=' * 60}")
    print(f"Total: {len(stems)} | OK: {len(stems) - len(failed)} | Failed: {len(failed)}")
    if failed:
        print("Failed experiments:")
        for s in failed:
            print(f"  {s}")
        sys.exit(1)


if __name__ == "__main__":
    main()
