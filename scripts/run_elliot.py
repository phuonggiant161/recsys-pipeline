"""
Run Elliot experiments from the repo root.

Usage:
    python scripts/run_elliot.py
    python scripts/run_elliot.py --filter hm_random_keep0.05
    python scripts/run_elliot.py --config hm_random_keep0.05_itemknn hm_random_keep0.05_vsm
    python scripts/run_elliot.py --filter hm_random_keep0.05 --dry-run
"""
import argparse
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
