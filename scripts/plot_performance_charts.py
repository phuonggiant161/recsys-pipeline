"""
Plot performance charts from the unified summary workbook.

Chart groups:
  random               — main metrics vs OSS/USS/ISS across random keep_fracs
  coldstart_u1         — Recall/nDCG_u1 vs OSS/USS/ISS
  coldstart_pop1       — Recall/nDCG_pop1 vs USS
  gini                 — main metrics vs item/user Gini
  beyond_accuracy      — coverage/popularity/diversity metrics
  coldstart_groups     — per-experiment categorical group charts +
                         popularity/user cross-variant charts (USS, ItemGini)
  group_distribution   — 100%-stacked item-popularity and user-activity bars

Usage:
    python scripts/plot_performance_charts.py
    python scripts/plot_performance_charts.py --input results/_summary/performance_summary.xlsx
    python scripts/plot_performance_charts.py --output-dir results/figures/performance_charts
    python scripts/plot_performance_charts.py --gini-keep-frac 0.1
    python scripts/plot_performance_charts.py --processed-dir data/processed
"""

import argparse
import csv
import re
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
MODEL_ORDER = ["VSM", "ItemKNN", "FunkSVD", "BPR", "NeuMF"]

_BASE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_LINESTYLES  = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
_MARKERS     = ["o", "s", "D", "^", "v", "P", "X", "*"]

MAIN_METRICS            = ["Precision", "Recall", "nDCG", "MAP", "MRR"]
COLD_U1_METRICS         = ["Recall_u1", "nDCG_u1"]
COLD_POP1_METRICS       = ["Recall_pop1", "nDCG_pop1"]
BEYOND_ACCURACY_METRICS = ["ItemCoverage", "AveragePopularity", "Gini", "TailPercentage"]
POP_GROUP_SUFFIXES      = ["pop1", "pop2_5", "pop6_10", "pop11_20", "pop21_40", "pop41plus"]
USER_GROUP_SUFFIXES     = ["u1", "u2_5", "u6_10", "u11_20", "u21_40", "u41plus"]
POP_GROUP_LABELS        = ["Pop 1", "Pop 2-5", "Pop 6-10", "Pop 11-20", "Pop 21-40", "Pop 41+"]
USER_GROUP_LABELS       = ["U 1", "U 2-5", "U 6-10", "U 11-20", "U 21-40", "U 41+"]
RANDOM_CUTOFFS          = [10, 20, 50]
GROUP_CUTOFF            = 20

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

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Pattern to parse experiment dir names for distribution charts
_EXP_DIST_PAT = re.compile(
    r"^(?P<dataset>.+?)_(?P<strategy>random|head|tail)_keep(?P<keep_frac>\d+(?:\.\d+)?)$"
)

# Popularity group bins (matches pipeline group spec)
_DIST_POP_BINS = [
    ("pop1",      lambda c: c == 1),
    ("pop2_5",    lambda c: 2 <= c <= 5),
    ("pop6_10",   lambda c: 6 <= c <= 10),
    ("pop11_20",  lambda c: 11 <= c <= 20),
    ("pop21_40",  lambda c: 21 <= c <= 40),
    ("pop41plus", lambda c: c >= 41),
]

# User activity group bins (matches pipeline group spec)
_DIST_USER_BINS = [
    ("u1",       lambda c: c == 1),
    ("u2_5",     lambda c: 2 <= c <= 5),
    ("u6_10",    lambda c: 6 <= c <= 10),
    ("u11_20",   lambda c: 11 <= c <= 20),
    ("u21_40",   lambda c: 21 <= c <= 40),
    ("u41plus",  lambda c: c >= 41),
]

# Sequential color palette for stacked bars (dark-red → orange → gold → green → blue → purple)
_DIST_GROUP_COLORS = [
    "#d62728", "#ff7f0e", "#e6c200",
    "#2ca02c", "#1f77b4", "#9467bd",
]

# Per-run cache: experiment name → distribution dict or None
_TRAIN_DIST_CACHE: dict = {}

# ── Path helper ───────────────────────────────────────────────────────────────

def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


# ── Stale PNG cleanup ─────────────────────────────────────────────────────────

def _clean_png_subtree(directory: Path) -> int:
    """Remove all *.png files recursively under *directory*.

    Called at the start of each chart-group generator to ensure that PNGs from
    previous runs (e.g. for metrics that have since been removed from the metric
    list) do not linger alongside newly generated charts.

    Returns the number of files removed.
    """
    count = 0
    if directory.exists():
        for png in directory.rglob("*.png"):
            try:
                png.unlink()
                count += 1
            except OSError:
                pass
    if count:
        print(f"  Cleaned {count} stale PNG(s) from {display_path(directory)}")
    return count


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
        _clean_png_subtree(out_base / ds / "random")
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
        _clean_png_subtree(out_base / ds / "coldstart_u1")
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
            for y_metric in COLD_U1_METRICS:
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


# ── Group 3: coldstart_pop1 ───────────────────────────────────────────────────

def generate_coldstart_pop1_charts(df: pd.DataFrame, out_base: Path, palette: dict) -> list[dict]:
    """
    Filter:  strategy=random, cutoff=20
    X-axis:  uss
    Y-axes:  COLD_POP1_METRICS (Recall_pop1, nDCG_pop1)
    Output:  {ds}/coldstart_pop1/cutoff_20/uss/{metric}_vs_uss.png
    Per dataset: 2 charts
    """
    CUTOFF  = GROUP_CUTOFF
    X_COL   = "uss"
    X_LABEL = "USS"
    index   = []
    for ds in sorted(df["dataset"].unique()):
        _clean_png_subtree(out_base / ds / "coldstart_pop1")
        sub_c = df[
            df["dataset"].eq(ds)
            & df["strategy"].eq("random")
            & df["cutoff"].eq(CUTOFF)
        ].copy()
        if sub_c.empty:
            continue
        ds_label = _fmt_dataset(ds)
        assert_no_duplicates(sub_c, ["experiment", "series_label", "cutoff"])
        labels = sorted_series_labels(sub_c["series_label"].unique().tolist())
        mixed  = check_mixed_frameworks(sub_c, f"coldstart_pop1/{ds}")

        for y_metric in COLD_POP1_METRICS:
            out_path = (
                out_base / ds / "coldstart_pop1"
                / f"cutoff_{CUTOFF}" / "uss"
                / f"{_metric_filename(y_metric)}_vs_uss.png"
            )
            title  = f"{ds_label} cold-start pop1: {y_metric}@{CUTOFF} vs {X_LABEL}"
            result = plot_single_metric_chart(
                sub=sub_c, x_col=X_COL, y_col=y_metric,
                title=title, x_label=X_LABEL, y_label=y_metric,
                palette=palette, out_path=out_path, labels=labels,
            )
            w = MIXED_FW_NOTE if mixed and result.get("status") == "ok" else ""
            index.append({
                "dataset":     ds,
                "chart_group": "coldstart_pop1",
                "strategy":    "random",
                "cutoff":      CUTOFF,
                "x_metric":   "uss",
                "y_metric":   y_metric,
                "gini_type":  "",
                "output_path": result.get("output_path", ""),
                "status":     result.get("status", "skip"),
                "reason":     result.get("reason", ""),
                "warning":    w,
            })
    return index


# ── Group 4 (old 3): gini ─────────────────────────────────────────────────────

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
        _clean_png_subtree(out_base / ds / "gini")
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


# ── Group 4: beyond_accuracy ──────────────────────────────────────────────────

def generate_beyond_accuracy_charts(df: pd.DataFrame, out_base: Path, palette: dict) -> list[dict]:
    """
    Filter:  strategy=random, cutoff=20
    X-axes:  oss (oss_pct), uss, iss
    Y-axes:  BEYOND_ACCURACY_METRICS (4)
    Output:  {ds}/beyond_accuracy/cutoff_20/{x_key}/{metric}_vs_{x_key}.png
    Per dataset: 3 x 4 = 12 charts
    """
    CUTOFF = GROUP_CUTOFF
    index  = []
    for ds in sorted(df["dataset"].unique()):
        _clean_png_subtree(out_base / ds / "beyond_accuracy")
        sub_c = df[
            df["dataset"].eq(ds)
            & df["strategy"].eq("random")
            & df["cutoff"].eq(CUTOFF)
        ].copy()
        if sub_c.empty:
            continue
        ds_label = _fmt_dataset(ds)
        assert_no_duplicates(sub_c, ["experiment", "series_label", "cutoff"])
        labels = sorted_series_labels(sub_c["series_label"].unique().tolist())
        mixed  = check_mixed_frameworks(sub_c, f"beyond_accuracy/{ds}")

        for x_key, (x_col, x_label) in X_AXIS_MAP.items():
            for y_metric in BEYOND_ACCURACY_METRICS:
                out_path = (
                    out_base / ds / "beyond_accuracy"
                    / f"cutoff_{CUTOFF}" / x_key
                    / f"{_metric_filename(y_metric)}_vs_{x_key}.png"
                )
                title  = f"{ds_label} beyond-accuracy: {y_metric}@{CUTOFF} vs {x_label}"
                result = plot_single_metric_chart(
                    sub=sub_c, x_col=x_col, y_col=y_metric,
                    title=title, x_label=x_label, y_label=y_metric,
                    palette=palette, out_path=out_path, labels=labels,
                )
                w = MIXED_FW_NOTE if mixed and result.get("status") == "ok" else ""
                index.append({
                    "dataset": ds, "chart_group": "beyond_accuracy",
                    "strategy": "random", "cutoff": CUTOFF,
                    "x_metric": x_key, "y_metric": y_metric,
                    "gini_type": "",
                    "output_path": result.get("output_path", ""),
                    "status": result.get("status", "skip"),
                    "reason": result.get("reason", ""),
                    "warning": w,
                })
    return index


# ── Train-distribution helpers ─────────────────────────────────────────────────

def _load_train_dist(experiment: str, processed_dir: Path) -> dict | None:
    """Load train.tsv and compute item/user group counts.

    Results are cached per experiment per run. TSV has no header: col 0 = user,
    col 1 = item. Returns None if train.tsv does not exist.
    """
    if experiment in _TRAIN_DIST_CACHE:
        return _TRAIN_DIST_CACHE[experiment]

    path = processed_dir / experiment / "train.tsv"
    if not path.exists():
        _TRAIN_DIST_CACHE[experiment] = None
        return None

    raw = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1], dtype=str)
    item_counts = raw[1].value_counts().to_dict()
    user_counts = raw[0].value_counts().to_dict()

    def _bin_counts(counts_dict, bins):
        return {name: sum(1 for c in counts_dict.values() if pred(c))
                for name, pred in bins}

    item_groups = _bin_counts(item_counts, _DIST_POP_BINS)
    user_groups = _bin_counts(user_counts, _DIST_USER_BINS)
    total_items = len(item_counts)
    total_users = len(user_counts)

    if sum(item_groups.values()) != total_items:
        warnings.warn(
            f"[dist/{experiment}] item group total {sum(item_groups.values())} "
            f"!= unique items {total_items}", stacklevel=2,
        )
    if sum(user_groups.values()) != total_users:
        warnings.warn(
            f"[dist/{experiment}] user group total {sum(user_groups.values())} "
            f"!= unique users {total_users}", stacklevel=2,
        )

    result = {
        "item_groups": item_groups,
        "user_groups": user_groups,
        "total_items": total_items,
        "total_users": total_users,
    }
    _TRAIN_DIST_CACHE[experiment] = result
    return result


def _plot_stacked_distribution(
    x_labels: list,
    percentages: list,
    raw_counts: list,
    bin_names: list,
    bin_display_labels: list,
    colors: list,
    title: str,
    y_label: str,
    out_path: Path,
) -> dict:
    """100%-stacked bar chart for group distributions.

    Each bar = one dataset variant; segments = group percentages.
    Annotates segment with percentage AND raw count when >= MIN_PCT_LABEL.
    Label format: "35.2%\\nn=3,482" (percentage on line 1, count on line 2).
    raw_counts[i][bin_name] must correspond to the i-th x_label.
    """
    if not x_labels:
        return {"status": "skip", "reason": "no experiments"}

    MIN_PCT_LABEL = 8.0
    bar_w = 0.65
    fig_w = max(5, len(x_labels) * 1.5 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.5))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Dataset variant")
    ax.set_ylabel(y_label)
    _light_grid(ax)

    x_pos    = list(range(len(x_labels)))
    bottoms  = [0.0] * len(x_labels)

    for bin_name, disp_label, color in zip(bin_names, bin_display_labels, colors):
        vals  = [pct.get(bin_name, 0.0) for pct in percentages]
        rects = ax.bar(x_pos, vals, bottom=bottoms, color=color,
                       label=disp_label, width=bar_w)
        for i, (rect, val, bot_val) in enumerate(zip(rects, vals, bottoms)):
            if val >= MIN_PCT_LABEL:
                cx = rect.get_x() + rect.get_width() / 2
                cy = bot_val + val / 2
                count = raw_counts[i].get(bin_name, 0)
                label_text = f"{val:.1f}%\nn={count:,}"
                ax.text(cx, cy, label_text, ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold",
                        linespacing=1.3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 108)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return {"status": "ok", "output_path": display_path(out_path)}


# ── Cross-variant charts: nDCG group vs x (USS or ItemGini) ───────────────────

def _generate_cross_variant_charts(
    df: pd.DataFrame,
    out_base: Path,
    palette: dict,
    cutoff: int,
    y_metrics: list,
    x_col: str,
    x_label: str,
    parent_folder: str,
    chart_group: str,
) -> list:
    """Cross-variant scatter/line charts: strategy=random, x=x_col, y=y_metric.

    Output: {ds}/coldstart_groups/{parent_folder}/cutoff_{cutoff}/{x_col}/{y_metric_lower}_vs_{x_col}.png
    """
    index = []
    for ds in sorted(df["dataset"].unique()):
        sub = df[
            df["dataset"].eq(ds)
            & df["strategy"].eq("random")
            & df["cutoff"].eq(cutoff)
        ].copy()
        if sub.empty:
            continue
        ds_label = _fmt_dataset(ds)
        assert_no_duplicates(sub, ["experiment", "series_label", "cutoff"])
        labels = sorted_series_labels(sub["series_label"].unique().tolist())
        mixed  = check_mixed_frameworks(sub, f"coldstart_groups/{ds}/{parent_folder}/{x_col}")

        for y_metric in y_metrics:
            out_path = (
                out_base / ds / "coldstart_groups" / parent_folder
                / f"cutoff_{cutoff}" / x_col
                / f"{_metric_filename(y_metric)}_vs_{x_col}.png"
            )
            title  = f"{ds_label}: {y_metric}@{cutoff} vs {x_label}"
            result = plot_single_metric_chart(
                sub=sub, x_col=x_col, y_col=y_metric,
                title=title, x_label=x_label, y_label=y_metric,
                palette=palette, out_path=out_path, labels=labels,
            )
            w = MIXED_FW_NOTE if mixed and result.get("status") == "ok" else ""
            index.append({
                "dataset":     ds,
                "chart_group": chart_group,
                "strategy":    "random",
                "cutoff":      cutoff,
                "x_metric":    x_col,
                "y_metric":    y_metric,
                "gini_type":   "",
                "output_path": result.get("output_path", ""),
                "status":      result.get("status", "skip"),
                "reason":      result.get("reason", ""),
                "warning":     w,
            })
    return index


# ── Group 5: coldstart_groups (categorical x-axis) ────────────────────────────

def plot_group_chart(
    sub: pd.DataFrame,
    group_suffixes: list[str],
    group_labels: list[str],
    y_col_template: str,
    title: str,
    y_label: str,
    palette: dict,
    out_path: Path,
    labels: list[str],
) -> dict:
    """Plot categorical group chart: x=group labels, one line per model.
    y_col_template uses positional format, e.g. 'Recall_{0}' → 'Recall_pop1'."""
    col_names  = [y_col_template.format(s) for s in group_suffixes]
    available  = [c for c in col_names if c in sub.columns and not sub[c].isna().all()]
    if not available:
        return {"status": "skip", "reason": f"No group columns found: {col_names}"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Group")
    ax.set_ylabel(y_label)
    _light_grid(ax)

    x_pos      = list(range(len(group_suffixes)))
    plotted_any = False
    for i, label in enumerate(labels):
        lsub = sub[sub["series_label"] == label]
        if lsub.empty:
            continue
        row = lsub.iloc[0]
        y_vals = [
            float(row[c]) if (c in sub.columns and pd.notna(row.get(c))) else None
            for c in col_names
        ]
        xp = [x_pos[j] for j, v in enumerate(y_vals) if v is not None]
        yp = [v for v in y_vals if v is not None]
        if not xp:
            warnings.warn(f"[{title}] '{label}' has no group data — skipped", stacklevel=2)
            continue
        color = palette.get(label, _BASE_COLORS[i % len(_BASE_COLORS)])
        ax.plot(
            xp, yp, label=label, color=color,
            linestyle=_LINESTYLES[i % len(_LINESTYLES)],
            marker=_MARKERS[i % len(_MARKERS)],
            linewidth=1.5, markersize=5,
        )
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return {"status": "skip", "reason": "no plottable data"}

    ax.set_xticks(x_pos)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
    ax.legend(loc="best", framealpha=0.9, fontsize=8)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    return {"status": "ok", "output_path": display_path(out_path)}


def generate_coldstart_groups_charts(df: pd.DataFrame, out_base: Path, palette: dict) -> list[dict]:
    """
    Per-experiment charts showing model performance across popularity/user groups.
    X-axis: categorical groups; lines: models; cutoff=20 only.
    Output: {ds}/coldstart_groups/popularity/{experiment}/recall.png  (and ndcg.png)
            {ds}/coldstart_groups/user/{experiment}/recall.png        (and ndcg.png)
    """
    CUTOFF = GROUP_CUTOFF
    index  = []

    # Clean per-dataset coldstart_groups subtrees before regenerating.
    for ds in sorted(df["dataset"].unique()):
        _clean_png_subtree(out_base / ds / "coldstart_groups")

    for exp in sorted(df["experiment"].unique()):
        exp_sub = df[df["experiment"].eq(exp) & df["cutoff"].eq(CUTOFF)].copy()
        if exp_sub.empty:
            continue
        ds       = exp_sub["dataset"].iloc[0]
        ds_label = _fmt_dataset(ds)
        assert_no_duplicates(exp_sub, ["series_label", "cutoff"])
        labels = sorted_series_labels(exp_sub["series_label"].unique().tolist())

        for group_type, suffixes, grp_labels, group_name in [
            ("popularity", POP_GROUP_SUFFIXES, POP_GROUP_LABELS, "Popularity"),
            ("user",       USER_GROUP_SUFFIXES, USER_GROUP_LABELS, "User"),
        ]:
            for y_prefix, y_label in [("Recall", f"Recall@{CUTOFF}"), ("nDCG", f"nDCG@{CUTOFF}")]:
                tmpl     = f"{y_prefix}_{{0}}"
                out_path = (
                    out_base / ds / "coldstart_groups" / group_type / exp
                    / f"{y_prefix.lower()}.png"
                )
                title = (
                    f"{ds_label} {exp}: {group_name} groups — {y_label}"
                )
                result = plot_group_chart(
                    sub=exp_sub, group_suffixes=suffixes, group_labels=grp_labels,
                    y_col_template=tmpl, title=title, y_label=y_label,
                    palette=palette, out_path=out_path, labels=labels,
                )
                strat = exp_sub["strategy"].iloc[0] if "strategy" in exp_sub.columns else ""
                index.append({
                    "dataset": ds, "chart_group": f"coldstart_{group_type}",
                    "strategy": strat, "cutoff": CUTOFF,
                    "x_metric": group_type, "y_metric": f"{y_prefix}@{CUTOFF}",
                    "gini_type": "",
                    "output_path": result.get("output_path", ""),
                    "status": result.get("status", "skip"),
                    "reason": result.get("reason", ""),
                    "warning": "",
                })

    # Steps 3-5: cross-variant charts (all run under the same coldstart_groups cleanup)
    index.extend(_generate_cross_variant_charts(
        df, out_base, palette, cutoff=CUTOFF,
        y_metrics=["nDCG_pop1", "nDCG_pop2_5"],
        x_col="uss", x_label="AIU (USS)",
        parent_folder="popularity",
        chart_group="coldstart_popularity_uss",
    ))

    index.extend(_generate_cross_variant_charts(
        df, out_base, palette, cutoff=CUTOFF,
        y_metrics=["nDCG_pop1", "nDCG_pop2_5"],
        x_col="item_gini", x_label="Item Gini",
        parent_folder="popularity",
        chart_group="coldstart_popularity_item_gini",
    ))

    index.extend(_generate_cross_variant_charts(
        df, out_base, palette, cutoff=CUTOFF,
        y_metrics=["nDCG_u2_5"],
        x_col="uss", x_label="AIU (USS)",
        parent_folder="user",
        chart_group="coldstart_user_uss",
    ))

    return index


# ── Group distribution charts ─────────────────────────────────────────────────

def generate_group_distribution_charts(out_base: Path, processed_dir: Path) -> list:
    """
    Discover experiment dirs under processed_dir, load train.tsv (cached), and
    plot 100%-stacked bar charts of item-popularity and user-activity groups.

    One chart per dataset × strategy × group_type.

    Output:
        {ds}/group_distribution/item_popularity/item_group_percentage_{strategy}.png
        {ds}/group_distribution/user_activity/user_group_percentage_{strategy}.png

    Experiments are discovered dynamically — no keep_fracs are hardcoded.
    """
    STRATEGIES = ["random", "head", "tail"]

    # Discover experiments from processed_dir
    all_exps: list = []
    for d in sorted(processed_dir.iterdir()):
        if not d.is_dir() or not (d / "train.tsv").exists():
            continue
        m = _EXP_DIST_PAT.match(d.name)
        if m:
            all_exps.append({
                "experiment": d.name,
                "dataset":    m.group("dataset"),
                "strategy":   m.group("strategy"),
                "keep_frac":  float(m.group("keep_frac")),
            })

    if not all_exps:
        print("  [WARN] No suitable experiment dirs found for distribution charts")
        return []

    datasets = sorted({e["dataset"] for e in all_exps})
    index: list = []

    for ds in datasets:
        _clean_png_subtree(out_base / ds / "group_distribution")
        ds_label = _fmt_dataset(ds)

        for strategy in STRATEGIES:
            exps = sorted(
                [e for e in all_exps if e["dataset"] == ds and e["strategy"] == strategy],
                key=lambda e: e["keep_frac"],
            )
            if not exps:
                continue

            # Load distributions (cached)
            dist_data = []
            for e in exps:
                d = _load_train_dist(e["experiment"], processed_dir)
                if d is None:
                    warnings.warn(
                        f"[dist] train.tsv missing for {e['experiment']}, skipped",
                        stacklevel=2,
                    )
                dist_data.append(d)

            valid_mask = [d is not None for d in dist_data]
            x_labels   = [
                f"keep {e['keep_frac']}" for e, ok in zip(exps, valid_mask) if ok
            ]
            if not x_labels:
                continue
            valid_dist = [d for d, ok in zip(dist_data, valid_mask) if ok]

            # ── Item popularity chart ──────────────────────────────────────
            pct_items   = []
            count_items = []
            for dist in valid_dist:
                total = dist["total_items"]
                pct_items.append({
                    k: (v / total * 100) if total > 0 else 0.0
                    for k, v in dist["item_groups"].items()
                })
                count_items.append(dist["item_groups"])

            out_path = (
                out_base / ds / "group_distribution" / "item_popularity"
                / f"item_group_percentage_{strategy}.png"
            )
            result = _plot_stacked_distribution(
                x_labels=x_labels,
                percentages=pct_items,
                raw_counts=count_items,
                bin_names=[b[0] for b in _DIST_POP_BINS],
                bin_display_labels=POP_GROUP_LABELS,
                colors=_DIST_GROUP_COLORS,
                title=f"{ds_label} {strategy}: item popularity group distribution",
                y_label="Percentage of items (%)",
                out_path=out_path,
            )
            index.append({
                "dataset":     ds,
                "chart_group": "item_group_distribution",
                "strategy":    strategy,
                "cutoff":      0,
                "x_metric":    "experiment",
                "y_metric":    "item_group_pct",
                "gini_type":   "",
                "output_path": result.get("output_path", ""),
                "status":      result.get("status", "skip"),
                "reason":      result.get("reason", ""),
                "warning":     "",
            })

            # ── User activity chart ────────────────────────────────────────
            pct_users   = []
            count_users = []
            for dist in valid_dist:
                total = dist["total_users"]
                pct_users.append({
                    k: (v / total * 100) if total > 0 else 0.0
                    for k, v in dist["user_groups"].items()
                })
                count_users.append(dist["user_groups"])

            out_path = (
                out_base / ds / "group_distribution" / "user_activity"
                / f"user_group_percentage_{strategy}.png"
            )
            result = _plot_stacked_distribution(
                x_labels=x_labels,
                percentages=pct_users,
                raw_counts=count_users,
                bin_names=[b[0] for b in _DIST_USER_BINS],
                bin_display_labels=USER_GROUP_LABELS,
                colors=_DIST_GROUP_COLORS,
                title=f"{ds_label} {strategy}: user activity group distribution",
                y_label="Percentage of users (%)",
                out_path=out_path,
            )
            index.append({
                "dataset":     ds,
                "chart_group": "user_group_distribution",
                "strategy":    strategy,
                "cutoff":      0,
                "x_metric":    "experiment",
                "y_metric":    "user_group_pct",
                "gini_type":   "",
                "output_path": result.get("output_path", ""),
                "status":      result.get("status", "skip"),
                "reason":      result.get("reason", ""),
                "warning":     "",
            })

    return index


def write_distribution_csv(out_base: Path, processed_dir: Path) -> None:
    """Write group-distribution counts and percentages to a CSV file.

    Columns: dataset, experiment, strategy, keep_frac, group_type, group, count, percentage.
    Covers all experiments discovered by _EXP_DIST_PAT (same set as the charts).
    Small-group counts that are not labeled on the chart remain inspectable here.
    """
    csv_path = out_base / "group_distribution_counts.csv"
    rows = []

    for d in sorted(processed_dir.iterdir()):
        if not d.is_dir():
            continue
        m = _EXP_DIST_PAT.match(d.name)
        if not m:
            continue
        exp      = d.name
        dataset  = m.group("dataset")
        strategy = m.group("strategy")
        kf       = float(m.group("keep_frac"))

        dist = _load_train_dist(exp, processed_dir)
        if dist is None:
            continue

        total_items = dist["total_items"]
        total_users = dist["total_users"]

        for bin_name, _ in _DIST_POP_BINS:
            count = dist["item_groups"].get(bin_name, 0)
            pct   = (count / total_items * 100) if total_items > 0 else 0.0
            rows.append({
                "dataset": dataset, "experiment": exp, "strategy": strategy,
                "keep_frac": kf, "group_type": "item_popularity",
                "group": bin_name, "count": count, "percentage": round(pct, 4),
            })

        for bin_name, _ in _DIST_USER_BINS:
            count = dist["user_groups"].get(bin_name, 0)
            pct   = (count / total_users * 100) if total_users > 0 else 0.0
            rows.append({
                "dataset": dataset, "experiment": exp, "strategy": strategy,
                "keep_frac": kf, "group_type": "user_activity",
                "group": bin_name, "count": count, "percentage": round(pct, 4),
            })

    if not rows:
        return

    fields = ["dataset", "experiment", "strategy", "keep_frac",
              "group_type", "group", "count", "percentage"]
    out_base.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Distribution counts CSV: {csv_path}")


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
        description="Plot performance charts (random, coldstart, gini, beyond_accuracy, coldstart_groups, group_distribution)"
    )
    p.add_argument("--input",          default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir",     default=str(DEFAULT_OUTDIR))
    p.add_argument("--processed-dir",  default=str(PROCESSED_DIR),
                   help="Path to data/processed/ for group-distribution charts")
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

    print("Generating coldstart_pop1 charts ...")
    index_pop1 = generate_coldstart_pop1_charts(df, out_base, palette)

    print(f"Generating gini charts (keep_frac={args.gini_keep_frac}) ...")
    index_gini = generate_gini_charts(df, out_base, palette, gini_keep_frac=args.gini_keep_frac)

    print("Generating beyond-accuracy charts ...")
    index_beyond = generate_beyond_accuracy_charts(df, out_base, palette)

    print("Generating coldstart group charts (per-experiment + cross-variant) ...")
    index_groups = generate_coldstart_groups_charts(df, out_base, palette)

    print("Generating group distribution charts ...")
    index_dist = generate_group_distribution_charts(out_base, Path(args.processed_dir))
    write_distribution_csv(out_base, Path(args.processed_dir))

    index = (index_random + index_cold + index_pop1 + index_gini
             + index_beyond + index_groups + index_dist)

    ok   = sum(1 for r in index if r.get("status") == "ok")
    skip = sum(1 for r in index if r.get("status") == "skip")
    print(f"\nFigures: {ok} generated, {skip} skipped")

    write_chart_index(index, out_base)
    write_generation_log(index, out_base, input_path)


if __name__ == "__main__":
    main()
