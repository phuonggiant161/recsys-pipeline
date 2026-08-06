"""
Plot performance charts from the unified summary workbook.
3 analysis groups: random, coldstart_u1, gini.

Usage:
    python scripts/plot_performance_charts.py
    python scripts/plot_performance_charts.py --input results/_summary/performance_summary.xlsx
    python scripts/plot_performance_charts.py --output-dir results/figures/performance_charts
    python scripts/plot_performance_charts.py --gini-keep-frac 0.1
"""

import argparse
import csv
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         9,
        "axes.titlesize":    10,
        "axes.labelsize":    9,
        "legend.fontsize":   8,
        "legend.framealpha": 0.9,
        "figure.dpi":        150,
    }
)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_INPUT  = PROJECT_ROOT / "results" / "_summary" / "performance_summary.xlsx"
DEFAULT_OUTDIR = PROJECT_ROOT / "results" / "figures" / "performance_charts"

# ── Constants ─────────────────────────────────────────────────────────────────

SAVE_DPI    = 300
MODEL_ORDER = ["VSM", "ItemKNN", "FunkSVD", "BPR"]

_BASE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_LINESTYLES  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_MARKERS     = ["o", "s", "D", "^", "v", "P", "X", "*"]

MAIN_METRICS   = ["Precision", "Recall", "nDCG", "MAP", "MRR"]
U1_METRICS     = ["Precision_u1", "Recall_u1", "nDCG_u1", "MAP_u1", "MRR_u1"]
RANDOM_CUTOFFS = [10, 20, 50]

# x_key -> (df_column, display_label)
X_AXIS_MAP = {
    "oss": ("oss_pct", "OSS (%)"),
    "uss": ("uss",     "USS"),
    "iss": ("iss",     "ISS"),
}
GINI_X_AXIS_MAP = {
    "item_gini": ("item_gini", "Item Gini"),
    "user_gini": ("user_gini", "User Gini"),
}

MIXED_FW_NOTE = (
    "Note: models are from different evaluation frameworks -- "
    "verify evaluator equivalence before cross-model comparison."
)

# ── Path helper ───────────────────────────────────────────────────────────────

def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


# ── Dataset display name ──────────────────────────────────────────────────────

def _fmt_dataset(name: str) -> str:
    return name.upper() if len(name) <= 3 else name.title()


# ── Metric filename component ─────────────────────────────────────────────────

def _metric_filename(metric: str) -> str:
    return metric.lower()


# ── Palette helpers ───────────────────────────────────────────────────────────

def sorted_series_labels(labels: list[str]) -> list[str]:
    return sorted(labels, key=lambda s: next((i for i, m in enumerate(MODEL_ORDER) if m in s), 99))


def make_palette(series_labels: list[str]) -> dict[str, str]:
    ordered = sorted_series_labels(list(series_labels))
    return {lbl: _BASE_COLORS[i % len(_BASE_COLORS)] for i, lbl in enumerate(ordered)}


def palette_for_df(df: pd.DataFrame) -> dict[str, str]:
    return make_palette(df["series_label"].unique().tolist())


# ── Grid helper ───────────────────────────────────────────────────────────────

def _light_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linewidth=0.6)


# ── Duplicate guard ───────────────────────────────────────────────────────────

def assert_no_duplicates(df: pd.DataFrame, key_cols: list[str]):
    dups = df[df.duplicated(subset=key_cols, keep=False)]
    if not dups.empty:
        preview = dups[key_cols].drop_duplicates().head(10).to_string(index=False)
        raise RuntimeError(
            f"Duplicate rows on {key_cols}:\n{preview}\n"
            "Run build_performance_summary.py first -- duplicates must be resolved there."
        )


# ── Float-safe keep_frac filter ───────────────────────────────────────────────

def _keep_frac_mask(df: pd.DataFrame, target: float) -> pd.Series:
    return np.isclose(df["keep_frac"].astype(float), target, rtol=0, atol=1e-9)


# ── Metric completeness check ─────────────────────────────────────────────────

def _check_metric_completeness(sub: pd.DataFrame, metrics: list[str], context: str) -> tuple[list[str], str]:
    missing = [m for m in metrics if m not in sub.columns or sub[m].isna().all()]
    if missing:
        msg = f"Metrics absent or all-NaN: {', '.join(missing)}"
        warnings.warn(f"[{context}] {msg}", stacklevel=3)
        return missing, msg
    return [], ""


# ── Mixed-framework warning ───────────────────────────────────────────────────

def check_mixed_frameworks(df: pd.DataFrame, context: str = "") -> bool:
    if df["framework"].nunique() > 1:
        fws = sorted(df["framework"].unique())
        warnings.warn(
            f"[Mixed frameworks] {context} -- frameworks present: {fws}. " + MIXED_FW_NOTE,
            stacklevel=3,
        )
        return True
    return False


# ── Load data ─────────────────────────────────────────────────────────────────

def load_long(input_path: Path) -> pd.DataFrame:
    df = pd.read_excel(input_path, sheet_name="long_all_results", engine="openpyxl")
    required = {"experiment", "dataset", "strategy", "keep_frac",
                "framework", "model", "series_label", "cutoff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"long_all_results sheet missing columns: {missing}")
    df["cutoff"]    = df["cutoff"].astype(int)
    df["keep_frac"] = df["keep_frac"].astype(float)
    return df


# ── Core single-chart function ────────────────────────────────────────────────

def plot_single_metric_chart(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    palette: dict,
    out_path: Path,
    labels: list[str],
    annotate_strategy: bool = False,
) -> dict:
    """
    Plot a single axes with one line per series_label (x_col vs y_col).
    Returns {"status": "ok"|"skip", "output_path": ..., "reason": ...}.
    """
    if y_col not in sub.columns or sub[y_col].isna().all():
        return {"status": "skip", "reason": f"{y_col} absent or all-NaN"}
    if x_col not in sub.columns or sub[x_col].isna().all():
        return {"status": "skip", "reason": f"{x_col} absent or all-NaN"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _light_grid(ax)

    plotted_any = False
    for i, label in enumerate(labels):
        lsub = sub[sub["series_label"] == label].dropna(subset=[x_col, y_col])
        if lsub.empty:
            continue
        lsub = lsub.sort_values(x_col)
        color = palette.get(label, _BASE_COLORS[i % len(_BASE_COLORS)])
        ax.plot(
            lsub[x_col], lsub[y_col],
            label=label, color=color,
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            marker=_MARKERS[i % len(_MARKERS)],
            linewidth=1.5, markersize=5,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return {"status": "skip", "reason": "no plottable data after dropna"}

    # Y-axis: force start at 0 unless max value is very small (chart would be too flat)
    y_max = sub[y_col].max(skipna=True)
    if pd.notna(y_max) and y_max > 0.02:
        ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))

    # Gini charts: annotate strategy names at each x position
    if annotate_strategy and "strategy" in sub.columns:
        strat_x = (
            sub.dropna(subset=[x_col])
               .groupby("strategy")[x_col].first()
               .sort_values()
        )
        y_bottom = ax.get_ylim()[0]
        for strat, x_val in strat_x.items():
            ax.annotate(
                strat,
                xy=(x_val, y_bottom),
                xytext=(0, -18),
                textcoords="offset points",
                ha="center", va="top",
                fontsize=7, color="#333333",
                annotation_clip=False,
            )

    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

    return {"status": "ok", "output_path": display_path(out_path)}


# ── Group 1: random ───────────────────────────────────────────────────────────

def generate_random_charts(df: pd.DataFrame, out_base: Path, palette: dict) -> list[dict]:
    """
    Filter:  strategy=random
    Cutoffs: 10, 20, 50
    X-axes:  oss (oss_pct), uss, iss
    Y-axes:  MAIN_METRICS (5)
    Output:  {ds}/random/cutoff_{N}/{x_key}/{y_metric}_vs_{x_key}.png
    Per dataset: 3 x 3 x 5 = 45 charts
    """
    index = []
    for ds in sorted(df["dataset"].unique()):
        ds_sub = df[df["dataset"].eq(ds) & df["strategy"].eq("random")].copy()
        if ds_sub.empty:
            continue
        ds_label = _fmt_dataset(ds)

        for cutoff in RANDOM_CUTOFFS:
            sub_c = ds_sub[ds_sub["cutoff"].eq(cutoff)].copy()
            if sub_c.empty:
                continue

            assert_no_duplicates(sub_c, ["experiment", "series_label", "cutoff"])
            labels = sorted_series_labels(sub_c["series_label"].unique().tolist())
            mixed  = check_mixed_frameworks(sub_c, f"random/{ds}/cutoff{cutoff}")

            for x_key, (x_col, x_label) in X_AXIS_MAP.items():
                for y_metric in MAIN_METRICS:
                    out_path = (
                        out_base / ds / "random"
                        / f"cutoff_{cutoff}" / x_key
                        / f"{_metric_filename(y_metric)}_vs_{x_key}.png"
                    )
                    title  = f"{ds_label} random cut: {y_metric}@{cutoff} vs {x_label}"
                    result = plot_single_metric_chart(
                        sub=sub_c, x_col=x_col, y_col=y_metric,
                        title=title, x_label=x_label, y_label=y_metric,
                        palette=palette, out_path=out_path, labels=labels,
                    )
                    w = MIXED_FW_NOTE if mixed and result.get("status") == "ok" else ""
                    index.append({
                        "dataset":     ds,
                        "chart_group": "random",
                        "strategy":    "random",
                        "cutoff":      cutoff,
                        "x_metric":   x_key,
                        "y_metric":   y_metric,
                        "gini_type":  "",
                        "output_path": result.get("output_path", ""),
                        "status":     result.get("status", "skip"),
                        "reason":     result.get("reason", ""),
                        "warning":    w,
                    })
    return index


# ── Group 2: coldstart_u1 ─────────────────────────────────────────────────────

def generate_coldstart_charts(df: pd.DataFrame, out_base: Path, palette: dict) -> list[dict]:
    """
    Filter:  strategy=random, cutoff=20
    X-axes:  oss (oss_pct), uss, iss
    Y-axes:  U1_METRICS (5)
    Output:  {ds}/coldstart_u1/cutoff_20/{x_key}/{y_metric}_vs_{x_key}.png
    Per dataset: 1 x 3 x 5 = 15 charts
    """
    COLDSTART_CUTOFF = 20
    index = []
    for ds in sorted(df["dataset"].unique()):
        sub_c = df[
            df["dataset"].eq(ds)
            & df["strategy"].eq("random")
            & df["cutoff"].eq(COLDSTART_CUTOFF)
        ].copy()
        if sub_c.empty:
            continue
        ds_label = _fmt_dataset(ds)

        assert_no_duplicates(sub_c, ["experiment", "series_label", "cutoff"])
        labels = sorted_series_labels(sub_c["series_label"].unique().tolist())
        mixed  = check_mixed_frameworks(sub_c, f"coldstart_u1/{ds}")

        for x_key, (x_col, x_label) in X_AXIS_MAP.items():
            for y_metric in U1_METRICS:
                out_path = (
                    out_base / ds / "coldstart_u1"
                    / f"cutoff_{COLDSTART_CUTOFF}" / x_key
                    / f"{_metric_filename(y_metric)}_vs_{x_key}.png"
                )
                title  = f"{ds_label} cold-start users: {y_metric}@{COLDSTART_CUTOFF} vs {x_label}"
                result = plot_single_metric_chart(
                    sub=sub_c, x_col=x_col, y_col=y_metric,
                    title=title, x_label=x_label, y_label=y_metric,
                    palette=palette, out_path=out_path, labels=labels,
                )
                w = MIXED_FW_NOTE if mixed and result.get("status") == "ok" else ""
                index.append({
                    "dataset":     ds,
                    "chart_group": "coldstart_u1",
                    "strategy":    "random",
                    "cutoff":      COLDSTART_CUTOFF,
                    "x_metric":   x_key,
                    "y_metric":   y_metric,
                    "gini_type":  "",
                    "output_path": result.get("output_path", ""),
                    "status":     result.get("status", "skip"),
                    "reason":     result.get("reason", ""),
                    "warning":    w,
                })
    return index


# ── Group 3: gini ─────────────────────────────────────────────────────────────

def generate_gini_charts(
    df: pd.DataFrame, out_base: Path, palette: dict, gini_keep_frac: float
) -> list[dict]:
    """
    Filter:  strategy in [head, random, tail], keep_frac~=gini_keep_frac
    Cutoffs: 10, 20, 50
    X-axes:  item_gini, user_gini
    Y-axes:  MAIN_METRICS (5)
    Output:  {ds}/gini/cutoff_{N}/{gini_key}/{y_metric}_vs_{gini_key}.png
    Per dataset: 3 x 2 x 5 = 30 charts
    """
    GINI_STRATEGIES = ["head", "random", "tail"]
    index = []
    for ds in sorted(df["dataset"].unique()):
        ds_sub = df[
            df["dataset"].eq(ds)
            & df["strategy"].isin(GINI_STRATEGIES)
            & _keep_frac_mask(df, gini_keep_frac)
        ].copy()

        if ds_sub.empty:
            for cutoff in RANDOM_CUTOFFS:
                for gini_key in GINI_X_AXIS_MAP:
                    for y_metric in MAIN_METRICS:
                        index.append({
                            "dataset":     ds,
                            "chart_group": "gini",
                            "strategy":    "head/random/tail",
                            "cutoff":      cutoff,
                            "x_metric":   gini_key,
                            "y_metric":   y_metric,
                            "gini_type":  gini_key,
                            "output_path": "",
                            "status":     "skip",
                            "reason":     f"no data for keep_frac={gini_keep_frac}",
                            "warning":    "",
                        })
            continue

        present_strategies = set(ds_sub["strategy"].unique())
        missing_strategies = set(GINI_STRATEGIES) - present_strategies
        if missing_strategies:
            reason = f"Missing strategies: {sorted(missing_strategies)}"
            warnings.warn(f"[gini/{ds}] {reason}", stacklevel=2)
            for cutoff in RANDOM_CUTOFFS:
                for gini_key in GINI_X_AXIS_MAP:
                    for y_metric in MAIN_METRICS:
                        index.append({
                            "dataset":     ds,
                            "chart_group": "gini",
                            "strategy":    "head/random/tail",
                            "cutoff":      cutoff,
                            "x_metric":   gini_key,
                            "y_metric":   y_metric,
                            "gini_type":  gini_key,
                            "output_path": "",
                            "status":     "skip",
                            "reason":     reason,
                            "warning":    "",
                        })
            continue

        ds_label = _fmt_dataset(ds)

        for cutoff in RANDOM_CUTOFFS:
            sub_c = ds_sub[ds_sub["cutoff"].eq(cutoff)].copy()
            if sub_c.empty:
                continue

            assert_no_duplicates(sub_c, ["experiment", "series_label", "cutoff"])
            labels = sorted_series_labels(sub_c["series_label"].unique().tolist())
            mixed  = check_mixed_frameworks(sub_c, f"gini/{ds}/cutoff{cutoff}")

            for gini_key, (gini_col, gini_label) in GINI_X_AXIS_MAP.items():
                for y_metric in MAIN_METRICS:
                    out_path = (
                        out_base / ds / "gini"
                        / f"cutoff_{cutoff}" / gini_key
                        / f"{_metric_filename(y_metric)}_vs_{gini_key}.png"
                    )
                    title = (
                        f"{ds_label} keep_frac={gini_keep_frac}: "
                        f"{y_metric}@{cutoff} vs {gini_label}"
                    )
                    result = plot_single_metric_chart(
                        sub=sub_c, x_col=gini_col, y_col=y_metric,
                        title=title, x_label=gini_label, y_label=y_metric,
                        palette=palette, out_path=out_path, labels=labels,
                        annotate_strategy=True,
                    )
                    w = MIXED_FW_NOTE if mixed and result.get("status") == "ok" else ""
                    index.append({
                        "dataset":     ds,
                        "chart_group": "gini",
                        "strategy":    "head/random/tail",
                        "cutoff":      cutoff,
                        "x_metric":   gini_key,
                        "y_metric":   y_metric,
                        "gini_type":  gini_key,
                        "output_path": result.get("output_path", ""),
                        "status":     result.get("status", "skip"),
                        "reason":     result.get("reason", ""),
                        "warning":    w,
                    })
    return index


# ── Outputs ───────────────────────────────────────────────────────────────────

_INDEX_FIELDS = [
    "dataset", "chart_group", "strategy", "cutoff",
    "x_metric", "y_metric", "gini_type",
    "output_path", "status", "reason", "warning",
]


def write_chart_index(index: list[dict], out_base: Path):
    path = out_base / "chart_index.csv"
    out_base.mkdir(parents=True, exist_ok=True)
    if not index:
        print("[WARN] No charts generated -- chart_index.csv will be empty.")
        path.write_text(",".join(_INDEX_FIELDS) + "\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(index)
    print(f"Chart index: {path}")


def write_generation_log(index: list[dict], out_base: Path, input_path: Path):
    path  = out_base / "chart_generation_log.txt"
    ok    = sum(1 for r in index if r.get("status") == "ok")
    skip  = sum(1 for r in index if r.get("status") == "skip")
    error = sum(1 for r in index if r.get("status") not in ("ok", "skip"))
    lines = [
        f"Input: {input_path}",
        f"Output directory: {out_base}",
        f"Total: {len(index)} | OK: {ok} | Skipped: {skip} | Error: {error}",
        "",
        "--- Details ---",
    ]
    for r in index:
        st  = r.get("status", "?")
        msg = r.get("reason", "") or r.get("warning", "")
        lines.append(
            f"[{st:5s}] {str(r.get('dataset','?')):12s} "
            f"{str(r.get('chart_group','?')):14s} "
            f"cut={str(r.get('cutoff','?')):3s} "
            f"{str(r.get('x_metric','?')):10s} "
            f"{str(r.get('y_metric','?')):15s} {msg}"
        )
    out_base.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generation log: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot performance charts (3 groups: random, coldstart_u1, gini)"
    )
    p.add_argument("--input",          default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir",     default=str(DEFAULT_OUTDIR))
    p.add_argument(
        "--gini-keep-frac", type=float, default=0.1,
        help="keep_frac filter for gini group (default 0.1)",
    )
    return p.parse_args()


def main():
    args       = parse_args()
    input_path = Path(args.input)
    out_base   = Path(args.output_dir)

    if not input_path.exists():
        sys.exit(
            f"ERROR: Input file not found: {input_path}\n"
            "Run build_performance_summary.py first."
        )

    print(f"Loading: {input_path}")
    df = load_long(input_path)
    print(
        f"  {len(df)} rows | {df['dataset'].nunique()} datasets | "
        f"{df['model'].nunique()} models | cutoffs {sorted(df['cutoff'].unique().tolist())}"
    )

    palette = palette_for_df(df)

    print("\nGenerating random charts ...")
    index_random = generate_random_charts(df, out_base, palette)

    print("Generating coldstart_u1 charts ...")
    index_cold = generate_coldstart_charts(df, out_base, palette)

    print(f"Generating gini charts (keep_frac={args.gini_keep_frac}) ...")
    index_gini = generate_gini_charts(df, out_base, palette, gini_keep_frac=args.gini_keep_frac)

    index = index_random + index_cold + index_gini

    ok   = sum(1 for r in index if r.get("status") == "ok")
    skip = sum(1 for r in index if r.get("status") == "skip")
    print(f"\nFigures: {ok} generated, {skip} skipped")

    write_chart_index(index, out_base)
    write_generation_log(index, out_base, input_path)


if __name__ == "__main__":
    main()
