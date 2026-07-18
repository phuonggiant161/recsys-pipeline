"""
Plot performance charts from Elliot experiment summary.

Groups:
  1. random_cut     — strategy=random (+base anchor), x=OSS/USS/ISS, y=main metrics, per cutoff
  2. coldstart_u1   — strategy=random, cutoff=20, x=OSS/USS/ISS, y=u1 metrics
  3. gini           — head/random/tail at keep_frac=0.5, x=item_gini/user_gini, y=main metrics

Usage:
    python scripts/plot_performance_charts.py
    python scripts/plot_performance_charts.py --input path/to/summary.xlsx --output path/to/figures
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "D:/recsys-pipeline/results/elliot/_summary/elliot_performance_summary.xlsx"
DEFAULT_OUTPUT = "D:/recsys-pipeline/results/figures/performance_charts"

MAIN_METRICS = ["Precision", "Recall", "nDCG", "MAP", "MRR"]
U1_METRICS   = ["Precision_u1", "Recall_u1", "nDCG_u1", "MAP_u1", "MRR_u1"]

# x-axis columns for groups 1 & 2
X_METRICS = [
    ("OSS", "oss_pct",  "OSS (%)"),
    ("USS", "uss",      "USS"),
    ("ISS", "iss",      "ISS"),
]

# gini columns for group 3
GINI_METRICS = [
    ("item_gini", "item_gini", "Item Gini"),
    ("user_gini", "user_gini", "User Gini"),
]

U1_CUTOFF      = 20
GINI_KEEP_FRAC = 0.1
GINI_STRATEGIES = ["head", "random", "tail"]

_DATASET_TITLE = {"hm": "HM", "amazon": "Amazon"}

DPI     = 300
FIGSIZE = (8.5, 5.2)

# Markers / colors cycle
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]
_COLORS  = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_fname(text):
    """Lowercase, replace non-alphanumeric (except _) with underscore."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def _title_dataset(dataset):
    return _DATASET_TITLE.get(str(dataset).lower(), str(dataset).title())


def load_performance_summary(input_path):
    """Load long_all_results sheet; convert coldstart pct cols to numeric."""
    df = pd.read_excel(input_path, sheet_name="long_all_results")
    for col in ["coldstart_user_pct", "coldstart_item_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in df.select_dtypes(include="object").columns:
        if col in ("dataset", "strategy", "model", "experiment", "full_model"):
            df[col] = df[col].astype(str).str.strip()
    return df


def save_chart(fig, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _draw_line_chart(sub_df, x_col, y_col, x_label, y_label, title, models,
                     annotate_strategy=False):
    """Draw one line-per-model chart. Returns fig or None if no plottable data."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    plotted = 0
    for i, model in enumerate(models):
        mdf = sub_df[sub_df["model"] == model].dropna(subset=[x_col, y_col])
        if mdf.empty:
            continue
        # aggregate duplicates (same x → mean y)
        group_cols = [x_col]
        if annotate_strategy and "strategy" in mdf.columns:
            group_cols.append("strategy")
        mdf = mdf.groupby(group_cols, as_index=False)[y_col].mean()
        mdf = mdf.sort_values(x_col)

        color  = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(mdf[x_col], mdf[y_col],
                marker=marker, color=color, label=model,
                linewidth=1.8, markersize=6)

        if annotate_strategy and "strategy" in mdf.columns:
            for _, r in mdf.iterrows():
                ax.annotate(str(r["strategy"]),
                            xy=(r[x_col], r[y_col]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=8, color=color)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=12)
    ax.legend(title="Model")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── Group 1: random cut ───────────────────────────────────────────────────────

def plot_metric_vs_sparsity(df, out_root, log_entries):
    """Group 1: strategy=random (+base anchor), x=OSS/USS/ISS, per cutoff."""
    data    = df[df["strategy"].isin(["random", "base"])].copy()
    datasets = sorted(data["dataset"].unique())
    cutoffs  = sorted(data["cutoff"].unique())
    models   = sorted(data["model"].unique())

    for dataset in datasets:
        dtitle = _title_dataset(dataset)
        for cutoff in cutoffs:
            sub = data[(data["dataset"] == dataset) & (data["cutoff"] == cutoff)]
            if sub.empty:
                continue
            for folder_name, x_col, x_label in X_METRICS:
                if x_col not in sub.columns:
                    continue
                for y_metric in MAIN_METRICS:
                    if y_metric not in sub.columns:
                        continue
                    if sub[y_metric].isna().all():
                        log_entries.append(_log_entry(
                            dataset, "random", "random", cutoff,
                            folder_name, y_metric, "",
                            "", f"skip: all {y_metric} null"
                        ))
                        continue

                    title  = f"{dtitle} random cut: {y_metric}@{cutoff} vs {x_label}"
                    y_label = f"{y_metric}@{cutoff}"
                    fig = _draw_line_chart(sub, x_col, y_metric, x_label, y_label, title, models)
                    if fig is None:
                        log_entries.append(_log_entry(
                            dataset, "random", "random", cutoff,
                            folder_name, y_metric, "",
                            "", "skip: no plottable data"
                        ))
                        continue

                    fname    = f"{_safe_fname(y_metric)}_vs_{_safe_fname(x_col).rstrip('_pct').replace('oss_','oss')}.png"
                    # Simplify: oss_pct → oss
                    fname    = re.sub(r"_pct$", "", fname.replace(".png", "")) + ".png"
                    out_path = out_root / dataset / "random" / f"cutoff_{cutoff}" / folder_name / fname

                    save_chart(fig, out_path)
                    log_entries.append(_log_entry(
                        dataset, "random", "random", cutoff,
                        folder_name, y_metric, "", str(out_path), "ok"
                    ))


# ── Group 2: cold-start u1 ────────────────────────────────────────────────────

def plot_coldstart_u1(df, out_root, log_entries):
    """Group 2: strategy=random, cutoff=20, x=OSS/USS/ISS, y=u1 metrics."""
    data    = df[df["strategy"].isin(["random", "base"])].copy()
    datasets = sorted(data["dataset"].unique())
    models   = sorted(data["model"].unique())

    for dataset in datasets:
        dtitle = _title_dataset(dataset)
        sub_c  = data[(data["dataset"] == dataset) & (data["cutoff"] == U1_CUTOFF)]
        if sub_c.empty:
            continue
        for folder_name, x_col, x_label in X_METRICS:
            if x_col not in sub_c.columns:
                continue
            for y_metric in U1_METRICS:
                if y_metric not in sub_c.columns:
                    continue
                if sub_c[y_metric].isna().all():
                    log_entries.append(_log_entry(
                        dataset, "coldstart_u1", "random", U1_CUTOFF,
                        folder_name, y_metric, "",
                        "", f"skip: all {y_metric} null"
                    ))
                    continue

                base_name = y_metric.replace("_u1", "")
                title     = f"{dtitle} cold-start users: {y_metric}@{U1_CUTOFF} vs {x_label}"
                y_label   = f"{y_metric}@{U1_CUTOFF}"
                fig = _draw_line_chart(sub_c, x_col, y_metric, x_label, y_label, title, models)
                if fig is None:
                    log_entries.append(_log_entry(
                        dataset, "coldstart_u1", "random", U1_CUTOFF,
                        folder_name, y_metric, "",
                        "", "skip: no plottable data"
                    ))
                    continue

                fname    = f"{_safe_fname(y_metric)}_vs_{folder_name.lower()}.png"
                out_path = out_root / dataset / "coldstart_u1" / f"cutoff_{U1_CUTOFF}" / folder_name / fname

                save_chart(fig, out_path)
                log_entries.append(_log_entry(
                    dataset, "coldstart_u1", "random", U1_CUTOFF,
                    folder_name, y_metric, "", str(out_path), "ok"
                ))


# ── Group 3: gini comparison ──────────────────────────────────────────────────

def plot_gini_comparison(df, out_root, log_entries):
    """Group 3: head/random/tail at keep_frac=0.5, x=item_gini/user_gini."""
    data = df[
        df["strategy"].isin(GINI_STRATEGIES)
        & (df["keep_frac"].round(6) == round(GINI_KEEP_FRAC, 6))
    ].copy()

    datasets = sorted(data["dataset"].unique())
    cutoffs  = sorted(data["cutoff"].unique())
    models   = sorted(data["model"].unique())

    for dataset in datasets:
        dtitle = _title_dataset(dataset)
        for cutoff in cutoffs:
            sub = data[(data["dataset"] == dataset) & (data["cutoff"] == cutoff)]
            if sub.empty:
                continue
            for folder_name, gini_col, gini_label in GINI_METRICS:
                if gini_col not in sub.columns or sub[gini_col].isna().all():
                    log_entries.append(_log_entry(
                        dataset, "gini", "head/random/tail", cutoff,
                        gini_col, "all_metrics", folder_name,
                        "", f"skip: {gini_col} all null"
                    ))
                    continue
                for y_metric in MAIN_METRICS:
                    if y_metric not in sub.columns:
                        continue
                    if sub[y_metric].isna().all():
                        log_entries.append(_log_entry(
                            dataset, "gini", "head/random/tail", cutoff,
                            gini_col, y_metric, folder_name,
                            "", f"skip: all {y_metric} null"
                        ))
                        continue

                    title   = f"{dtitle} keep_frac={GINI_KEEP_FRAC}: {y_metric}@{cutoff} vs {gini_label}"
                    y_label = f"{y_metric}@{cutoff}"
                    fig = _draw_line_chart(
                        sub, gini_col, y_metric, gini_label, y_label,
                        title, models, annotate_strategy=True
                    )
                    if fig is None:
                        log_entries.append(_log_entry(
                            dataset, "gini", "head/random/tail", cutoff,
                            gini_col, y_metric, folder_name,
                            "", "skip: no plottable data"
                        ))
                        continue

                    fname    = f"{_safe_fname(y_metric)}_vs_{folder_name}.png"
                    out_path = out_root / dataset / "gini" / f"cutoff_{cutoff}" / folder_name / fname

                    save_chart(fig, out_path)
                    log_entries.append(_log_entry(
                        dataset, "gini", "head/random/tail", cutoff,
                        gini_col, y_metric, folder_name, str(out_path), "ok"
                    ))


# ── Index + log ───────────────────────────────────────────────────────────────

def _log_entry(dataset, chart_group, strategy, cutoff,
               x_metric, y_metric, gini_type, output_path, status):
    return {
        "dataset":     dataset,
        "chart_group": chart_group,
        "strategy":    strategy,
        "cutoff":      cutoff,
        "x_metric":    x_metric,
        "y_metric":    y_metric,
        "gini_type":   gini_type,
        "output_path": output_path,
        "status":      status,
    }


def build_chart_index(log_entries, out_root):
    cols = ["dataset", "chart_group", "strategy", "cutoff",
            "x_metric", "y_metric", "gini_type", "output_path", "status"]
    index_df = pd.DataFrame(log_entries, columns=cols)
    index_df.to_csv(out_root / "chart_index.csv", index=False, encoding="utf-8-sig")

    ok    = (index_df["status"] == "ok").sum()
    skips = index_df[index_df["status"] != "ok"]

    lines = [
        f"Charts generated : {ok}",
        f"Charts skipped   : {len(skips)}",
        "",
    ]
    if not skips.empty:
        lines.append("Skipped charts:")
        for _, r in skips.iterrows():
            lines.append(
                f"  [{r['dataset']}] {r['chart_group']} cutoff={r['cutoff']} "
                f"x={r['x_metric']} y={r['y_metric']} — {r['status']}"
            )
    log_text = "\n".join(lines)
    (out_root / "chart_generation_log.txt").write_text(log_text, encoding="utf-8")
    print(log_text[:500])
    return index_df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Plot Elliot performance charts")
    p.add_argument("--input",  default=DEFAULT_INPUT)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    return p.parse_args()


def main():
    args     = parse_args()
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    df = load_performance_summary(args.input)
    print(f"Loaded {len(df)} rows | datasets={sorted(df['dataset'].unique())} | models={sorted(df['model'].unique())}")

    log_entries = []

    print("\nGroup 1: random cut charts ...")
    plot_metric_vs_sparsity(df, out_root, log_entries)

    print("Group 2: cold-start u1 charts ...")
    plot_coldstart_u1(df, out_root, log_entries)

    print("Group 3: gini comparison charts ...")
    plot_gini_comparison(df, out_root, log_entries)

    print("\nBuilding index and log ...")
    index_df = build_chart_index(log_entries, out_root)

    ok = (index_df["status"] == "ok").sum()
    print(f"\nDone. {ok} charts written to: {out_root}")
    print(f"  chart_index.csv           : {out_root / 'chart_index.csv'}")
    print(f"  chart_generation_log.txt  : {out_root / 'chart_generation_log.txt'}")


if __name__ == "__main__":
    main()
