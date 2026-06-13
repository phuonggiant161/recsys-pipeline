"""
Run Elliot experiments from the repo root.

Usage:
    # Train + evaluate (ItemKNN)
    python scripts/run_elliot.py
    python scripts/run_elliot.py --filter hm_k20
    python scripts/run_elliot.py --config hm_k20_dedup_base_split_itemknn hm_k20_random_keep0.5_itemknn
    python scripts/run_elliot.py --filter hm_k20 --dry-run

    # Eval-only (RecommendationFolder, requires existing rec files)
    python scripts/run_elliot.py --eval-only
    python scripts/run_elliot.py --eval-only --filter hm_k20
    python scripts/run_elliot.py --eval-only --config hm_k20_random_keep0.1_eval
    python scripts/run_elliot.py --eval-only --filter hm_k20 --dry-run

    Note: generate eval-only configs first with:
        python scripts/generate_elliot_configs.py --eval-only
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ELLIOT_DIR = PROJECT_ROOT / "external" / "elliot"
CONFIG_DIR = ELLIOT_DIR / "config_files"
EVAL_CONFIG_DIR = ELLIOT_DIR / "config_files_eval"


def get_python() -> str:
    """Use venv_elliot if it exists, otherwise fall back to current Python."""
    for candidate in [
        ELLIOT_DIR / "venv_elliot" / "Scripts" / "python.exe",  # Windows
        ELLIOT_DIR / "venv_elliot" / "bin" / "python",          # Linux/Mac
    ]:
        if candidate.exists():
            return str(candidate)
    return sys.executable


def collect_configs(filter_text: str, specific: list[str], eval_only: bool) -> list[str]:
    """Return list of config stems to run, in sorted order."""
    if specific:
        return sorted(specific)

    config_dir = EVAL_CONFIG_DIR if eval_only else CONFIG_DIR
    if not config_dir.exists():
        return []

    stems = sorted(p.stem for p in config_dir.glob("*.yml"))
    if filter_text:
        stems = [s for s in stems if filter_text in s]
    return stems


def run_one(stem: str, dry_run: bool, eval_only: bool) -> bool:
    """Run a single experiment. Returns True on success."""
    config_dir_name = "config_files_eval" if eval_only else "config_files"
    cmd = [get_python(), "start_experiments.py",
           "--config", stem,
           "--config-dir", config_dir_name]

    label = "[EVAL-ONLY]" if eval_only else "[TRAIN]"
    print(f"\n{'[DRY-RUN] ' if dry_run else ''}{label} {stem}")
    print(f"  config : {config_dir_name}/{stem}.yml")
    print(f"  cmd    : {' '.join(cmd)}")
    print(f"  cwd    : {ELLIOT_DIR}")

    if dry_run:
        return True

    result = subprocess.run(cmd, cwd=ELLIOT_DIR)
    if result.returncode != 0:
        print(f"[FAILED] {stem} (exit code {result.returncode})")
        return False

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
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Run eval-only configs from config_files_eval/ (RecommendationFolder). "
             "Generate them first with: python scripts/generate_elliot_configs.py --eval-only",
    )
    args = parser.parse_args()

    if args.eval_only and not EVAL_CONFIG_DIR.exists():
        print(f"[ERROR] Eval config directory not found: {EVAL_CONFIG_DIR}")
        print("        Run first: python scripts/generate_elliot_configs.py --eval-only")
        sys.exit(1)

    stems = collect_configs(filter_text=args.filter, specific=args.config, eval_only=args.eval_only)

    if not stems:
        mode = "eval-only" if args.eval_only else "train"
        print(f"No {mode} configs found.")
        if args.eval_only:
            print("Generate them first: python scripts/generate_elliot_configs.py --eval-only")
        else:
            print("Use --filter or --config to specify experiments.")
        sys.exit(1)

    mode_label = "EVAL-ONLY" if args.eval_only else "TRAIN"
    print(f"Mode: {mode_label} | Experiments to run ({len(stems)}):")
    for s in stems:
        print(f"  {s}")

    failed = []
    for stem in stems:
        ok = run_one(stem, dry_run=args.dry_run, eval_only=args.eval_only)
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
