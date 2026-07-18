"""
Build Elliot performance summary Excel from TSV result files.

Usage:
    python scripts/build_performance_summary.py
    python scripts/build_performance_summary.py --output path/to/output.xlsx
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_RESULTS_ROOT   = "D:/recsys-pipeline/results/elliot"
DEFAULT_PROCESSED_ROOT = "D:/recsys-pipeline/data/processed"
DEFAULT_REPORTS_ROOT   = "D:/recsys-pipeline/data/reports"
DEFAULT_OUTPUT         = "D:/recsys-pipeline/results/elliot/_summary/elliot_performance_summary.xlsx"

ROUND_DIGITS = 6

MAIN_METRICS = ["Precision", "Recall", "nDCG", "MAP", "MRR"]
U1_METRICS   = ["Precision_u1", "Recall_u1", "nDCG_u1", "MAP_u1", "MRR_u1"]
ALL_METRICS  = MAIN_METRICS + U1_METRICS

SPARSITY_COLS = [
    "n_interactions_x1000", "oss_pct", "uss", "iss",
    "user_gini", "item_gini",
    "coldstart_user_pct", "coldstart_item_pct",
]

_STRATEGY_ORDER = {"base": 0, "head": 1, "random": 2, "tail": 3}


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_experiment_name(name):
    m = re.match(r"^(\w+?)_(random|head|tail)_keep([\d.]+)$", name)
    if m:
        return {"dataset": m.group(1), "strategy": m.group(2), "keep_frac": float(m.group(3))}
    m = re.match(r"^(\w+?)_dedup_base_split$", name)
    if m:
        return {"dataset": m.group(1), "strategy": "base", "keep_frac": 1.0}
    parts = name.split("_")
    return {"dataset": parts[0], "strategy": None, "keep_frac": None}


def load_kcore(exp_name, processed_root):
    mj = processed_root / exp_name / "metadata.json"
    if mj.exists():
        try:
            data = json.loads(mj.read_text(encoding="utf-8"))
            return data.get("k_user"), data.get("k_item")
        except Exception:
            pass
    return None, None


# ── Step 1: collect TSV performance rows ─────────────────────────────────────

def collect_performance_rows(results_root, processed_root):
    rows = []
    for ds_dir in sorted(results_root.iterdir()):
        if not ds_dir.is_dir() or ds_dir.name.startswith("_"):
            continue
        perf_dir = ds_dir / "performance"
        if not perf_dir.exists():
            continue

        exp_meta = parse_experiment_name(ds_dir.name)
        k_user, k_item = load_kcore(ds_dir.name, processed_root)

        for tsv in sorted(perf_dir.glob("*_cutoff_*.tsv")):
            m = re.match(r"^(.+)_cutoff_(\d+)\.tsv$", tsv.name)
            if not m:
                continue
            model_short = m.group(1)
            cutoff      = int(m.group(2))

            try:
                df = pd.read_csv(tsv, sep="\t")
            except Exception as e:
                print(f"[WARN] Cannot read {tsv}: {e}")
                continue

            if df.empty or "model" not in df.columns:
                continue

            for _, row in df.iterrows():
                record = {
                    "experiment": ds_dir.name,
                    "dataset":    exp_meta["dataset"],
                    "strategy":   exp_meta["strategy"],
                    "keep_frac":  exp_meta["keep_frac"],
                    "k_user":     k_user,
                    "k_item":     k_item,
                    "cutoff":     cutoff,
                    "model":      model_short,
                    "full_model": str(row.get("model", "")),
                }
                for metric in ALL_METRICS:
                    record[metric] = (
                        round(float(row[metric]), ROUND_DIGITS)
                        if metric in row and pd.notna(row[metric])
                        else float("nan")
                    )
                rows.append(record)

    if not rows:
        raise RuntimeError(f"No performance TSVs found under {results_root}")

    long_df = pd.DataFrame(rows)
    long_df["_so"] = long_df["strategy"].map(_STRATEGY_ORDER).fillna(99)
    long_df = (
        long_df
        .sort_values(["dataset", "_so", "keep_frac", "model", "cutoff"])
        .drop(columns=["_so"])
        .reset_index(drop=True)
    )
    return long_df


# ── Step 2: load sparsity reports ────────────────────────────────────────────

def load_dataset_reports(reports_root):
    dfs = []
    for csv_path in sorted(reports_root.glob("*_summary.csv")):
        m = re.match(r"^(\w+)_(random|head|tail|base)_summary\.csv$", csv_path.name)
        if not m:
            print(f"[WARN] Skipping unrecognized: {csv_path.name}")
            continue
        dataset  = m.group(1)
        strategy = m.group(2)
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Cannot read {csv_path.name}: {e}")
            continue
        df["dataset"]   = dataset
        df["strategy"]  = strategy
        df["keep_frac"] = df["keep_frac"].astype(float)
        keep_cols = ["dataset", "strategy", "keep_frac"] + [
            c for c in SPARSITY_COLS if c in df.columns
        ]
        dfs.append(df[keep_cols])

    if not dfs:
        print("[WARN] No report files found")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Synthesize base rows: copy keep_frac=1.0 from any strategy
    base_rows = []
    for dataset, grp in combined.groupby("dataset"):
        row_10 = grp[grp["keep_frac"] == 1.0].head(1).copy()
        if row_10.empty:
            print(f"[WARN] No keep_frac=1.0 for dataset={dataset} — base not synthesized")
            continue
        row_10["strategy"] = "base"
        base_rows.append(row_10)

    if base_rows:
        combined = pd.concat(
            [combined, pd.concat(base_rows, ignore_index=True)],
            ignore_index=True,
        )

    combined = combined.drop_duplicates(
        subset=["dataset", "strategy", "keep_frac"], keep="last"
    )
    return combined.reset_index(drop=True)


# ── Step 3: merge ─────────────────────────────────────────────────────────────

def merge_reports(long_df, report_df):
    if report_df.empty:
        print("[WARN] report_df empty — no sparsity columns added")
        return long_df

    long_df  = long_df.copy()
    rep_copy = report_df.copy()
    long_df["_kf"]  = long_df["keep_frac"].round(6)
    rep_copy["_kf"] = rep_copy["keep_frac"].round(6)

    merged = long_df.merge(
        rep_copy.drop(columns=["keep_frac"]),
        on=["dataset", "strategy", "_kf"],
        how="left",
    ).drop(columns=["_kf"])

    first_col = next((c for c in SPARSITY_COLS if c in merged.columns), None)
    if first_col:
        unmatched = merged[merged[first_col].isna()]
        if not unmatched.empty:
            miss = unmatched[["experiment", "dataset", "strategy", "keep_frac"]].drop_duplicates()
            print(f"[WARN] {len(unmatched)} rows missing sparsity ({len(miss)} experiments):")
            print(miss.to_string(index=False))
        else:
            print(f"All {len(merged)} rows matched a sparsity report.")

    return merged


# ── Step 4: write Excel ───────────────────────────────────────────────────────

def write_excel(long_df, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sparsity_present = [c for c in SPARSITY_COLS if c in long_df.columns]
    meta_cols = [
        "experiment", "dataset", "strategy", "keep_frac", "k_user", "k_item",
    ] + sparsity_present

    # Sheet 1: long_all_results
    col_order = meta_cols + ["cutoff", "model", "full_model"] + ALL_METRICS
    sheet_long = long_df[[c for c in col_order if c in long_df.columns]].copy()

    # Sheet 2: wide_summary
    idx = ["experiment", "dataset", "strategy", "keep_frac", "k_user", "k_item", "model"]
    wide_df = long_df.pivot_table(
        index=idx, columns="cutoff", values=MAIN_METRICS, aggfunc="first"
    )
    wide_df.columns = [f"{m}@{c}" for m, c in wide_df.columns]
    wide_df = wide_df.reset_index()
    if sparsity_present:
        spe = (
            long_df[["experiment"] + sparsity_present]
            .drop_duplicates(subset=["experiment"])
            .set_index("experiment")
        )
        wide_df = wide_df.join(spe, on="experiment")
    cutoffs = sorted(long_df["cutoff"].unique())
    metric_cols = [f"{m}@{c}" for c in cutoffs for m in MAIN_METRICS]
    final_wide = idx + sparsity_present + [c for c in metric_cols if c in wide_df.columns]
    wide_df = wide_df[[c for c in final_wide if c in wide_df.columns]]
    wide_df["_so"] = wide_df["strategy"].map(_STRATEGY_ORDER).fillna(99)
    wide_df = (
        wide_df.sort_values(["dataset", "_so", "keep_frac", "model"])
        .drop(columns=["_so"]).reset_index(drop=True)
    )

    # Sheet 3: main_metrics
    sheet_main = long_df[
        [c for c in meta_cols + ["cutoff", "model"] if c in long_df.columns] + MAIN_METRICS
    ].copy()

    # Sheet 4: u1_metrics
    sheet_u1 = long_df[
        [c for c in meta_cols + ["cutoff", "model"] if c in long_df.columns] + U1_METRICS
    ].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for df, name in [
            (sheet_long, "long_all_results"),
            (wide_df,    "wide_summary"),
            (sheet_main, "main_metrics"),
            (sheet_u1,   "u1_metrics"),
        ]:
            df.to_excel(writer, sheet_name=name, index=False)
            ws = writer.sheets[name]
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)
            ws.freeze_panes = "A2"

    print(f"Written: {output_path}")
    print(f"  long_all_results : {sheet_long.shape[0]} × {sheet_long.shape[1]}")
    print(f"  wide_summary     : {wide_df.shape[0]} × {wide_df.shape[1]}")
    print(f"  main_metrics     : {sheet_main.shape[0]} × {sheet_main.shape[1]}")
    print(f"  u1_metrics       : {sheet_u1.shape[0]} × {sheet_u1.shape[1]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build Elliot performance summary Excel")
    p.add_argument("--results-root",   default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--processed-root", default=DEFAULT_PROCESSED_ROOT)
    p.add_argument("--reports-root",   default=DEFAULT_REPORTS_ROOT)
    p.add_argument("--output",         default=DEFAULT_OUTPUT)
    return p.parse_args()


def main():
    args = parse_args()
    results_root   = Path(args.results_root)
    processed_root = Path(args.processed_root)
    reports_root   = Path(args.reports_root)
    output_path    = Path(args.output)

    print(f"Collecting TSVs from: {results_root}")
    long_df = collect_performance_rows(results_root, processed_root)
    print(f"Collected {len(long_df)} rows | {long_df['experiment'].nunique()} experiments")
    print(f"  Datasets: {sorted(long_df['dataset'].unique())}")
    print(f"  Models  : {sorted(long_df['model'].unique())}")
    print(f"  Cutoffs : {sorted(long_df['cutoff'].unique())}")

    print(f"\nLoading reports from: {reports_root}")
    report_df = load_dataset_reports(reports_root)
    if not report_df.empty:
        print(f"Report rows: {len(report_df)} | strategies: {sorted(report_df['strategy'].unique())}")

    long_df = merge_reports(long_df, report_df)

    print(f"\nWriting: {output_path}")
    write_excel(long_df, output_path)


if __name__ == "__main__":
    main()
