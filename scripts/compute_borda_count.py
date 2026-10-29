"""
Compute Borda Count rankings for the thesis Results section.

Single input:  results/_summary/performance_summary.xlsx (sheet long_all_results)
Output root:   results/borda_count/{overall,coldstart_user,longtail_item}/

Three analyses, each producing 01_full_borda_count.xlsx, 02_borda_by_dataset.xlsx,
03_final_borda.xlsx:

  overall         -- strategy=random, metric=nDCG@20, ballot = dataset x AIU_level
  coldstart_user  -- strategy=random, metric=nDCG@20, ballot = dataset x keep_frac x user_group
  longtail_item   -- strategy=random, metric=nDCG@20, ballot = dataset x keep_frac x item_group

Design decision (see report): coldstart_user / longtail_item restrict to
strategy=="random" even though the spec's ballot definition does not mention
strategy explicitly. head/tail share keep_frac values with random (e.g. both
head_keep0.1 and random_keep0.1 exist), so grouping by (dataset, keep_frac,
group) alone would silently merge three different experiments into one
ballot and duplicate every model 2-3x -- directly violating the "no duplicate
model per ballot" validation rule. Restricting to random avoids this and
matches how the existing coldstart charts (plot_performance_charts.py) are
already scoped.

No academic conclusions are written by this script -- only data outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_XLSX   = PROJECT_ROOT / "results" / "_summary" / "performance_summary.xlsx"
OUTPUT_ROOT  = PROJECT_ROOT / "results" / "borda_count"

# ── Constants ─────────────────────────────────────────────────────────────────

N_ALGORITHMS   = 8
CUTOFF         = 20
METRIC_COL     = "nDCG"          # long_all_results: metric column name for the main (non-grouped) nDCG
EXPECTED_SUM   = sum(range(N_ALGORITHMS))  # 7+6+...+0 = 28

# user_group / item_group label -> source column in long_all_results
USER_GROUP_SRC_COL = {
    "U1":     "nDCG_u1",
    "U2_5":   "nDCG_u2_5",
    "U6_10":  "nDCG_u6_10",
    "U11_20": "nDCG_u11_20",
    "U21_40": "nDCG_u21_40",
    "U41+":   "nDCG_u41plus",
}
ITEM_GROUP_SRC_COL = {
    "Pop1":     "nDCG_pop1",
    "Pop2_5":   "nDCG_pop2_5",
    "Pop6_10":  "nDCG_pop6_10",
    "Pop11_20": "nDCG_pop11_20",
    "Pop21_40": "nDCG_pop21_40",
    "Pop41+":   "nDCG_pop41plus",
}
USER_GROUP_ORDER = list(USER_GROUP_SRC_COL.keys())
ITEM_GROUP_ORDER = list(ITEM_GROUP_SRC_COL.keys())

REQUIRED_LONG_COLS = {
    "experiment", "dataset", "strategy", "keep_frac", "model", "cutoff",
    "AIU", METRIC_COL,
    *USER_GROUP_SRC_COL.values(),
    *ITEM_GROUP_SRC_COL.values(),
}


# ── 0. Schema loading / validation ────────────────────────────────────────────

def load_long_all_results(xlsx_path: Path) -> pd.DataFrame:
    """Load long_all_results and hard-validate the columns this script depends on."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input workbook not found: {xlsx_path}")
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")
    if "long_all_results" not in xl.sheet_names:
        raise ValueError(
            f"Sheet 'long_all_results' not found. Available sheets: {xl.sheet_names}"
        )
    df = pd.read_excel(xlsx_path, sheet_name="long_all_results", engine="openpyxl")

    missing = REQUIRED_LONG_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"long_all_results is missing required columns: {sorted(missing)}\n"
            f"Actual columns: {list(df.columns)}"
        )

    df["cutoff"]    = df["cutoff"].astype(int)
    df["keep_frac"] = df["keep_frac"].astype(float)
    return df


def validate_input_assumptions(df: pd.DataFrame, issues: list[dict]) -> list[str]:
    """
    Cross-check the hard assumptions this script relies on. Appends to `issues`
    (for the validation_issues sheet) and returns the sorted list of actual
    model names (expected to be exactly 8).
    """
    models = sorted(df["model"].dropna().unique().tolist())
    if len(models) != N_ALGORITHMS:
        issues.append({
            "analysis": "GLOBAL", "ballot": "", "issue_type": "model_count_mismatch",
            "detail": f"Expected {N_ALGORITHMS} models, found {len(models)}: {models}",
        })

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    for ds in ("hm", "amazon"):
        if ds not in datasets:
            issues.append({
                "analysis": "GLOBAL", "ballot": "", "issue_type": "dataset_missing",
                "detail": f"Expected dataset '{ds}' not found. Datasets present: {datasets}",
            })

    if CUTOFF not in set(df["cutoff"].unique().tolist()):
        issues.append({
            "analysis": "GLOBAL", "ballot": "", "issue_type": "cutoff_missing",
            "detail": f"cutoff={CUTOFF} not present in data. Cutoffs present: {sorted(df['cutoff'].unique().tolist())}",
        })

    return models


# ── 1. Core Borda ranking (generic, reused by all 3 analyses) ────────────────

def rank_and_borda(
    ballot_df: pd.DataFrame,
    score_col: str,
    model_col: str = "model",
    n_algorithms: int = N_ALGORITHMS,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Rank one ballot's models by score_col (descending) and compute Borda scores.

    Ties are handled via average rank (pandas rank(method="average")), then
    borda = n_algorithms - average_rank. This matches the spec's formula
    exactly: rank 1 -> n-1, rank n -> 0, tied ranks -> averaged borda.

    Rows with NaN score_col are dropped before ranking (a model that has no
    valid metric value cannot be ranked in this ballot). Returns the ranked
    sub-dataframe (with new "rank" and "borda_score" columns) plus a list of
    warning strings (e.g. duplicate model after dropna).

    Does NOT force ballots to have exactly n_algorithms rows -- callers must
    check completeness separately (see validate_ballot_completeness).
    """
    warnings: list[str] = []
    clean = ballot_df.dropna(subset=[score_col]).copy()

    dup = clean[clean.duplicated(subset=[model_col], keep=False)]
    if not dup.empty:
        dup_models = sorted(dup[model_col].unique().tolist())
        warnings.append(f"Duplicate model(s) in ballot after dropping NaN: {dup_models}")
        # Keep only the first occurrence per model to avoid corrupting the rank
        # -- this is a data-integrity problem the caller must see (returned in
        # warnings), not something to silently average away.
        clean = clean.drop_duplicates(subset=[model_col], keep="first")

    if clean.empty:
        return clean.assign(rank=pd.Series(dtype=float), borda_score=pd.Series(dtype=float)), warnings

    clean["rank"] = clean[score_col].rank(method="average", ascending=False)
    clean["borda_score"] = n_algorithms - clean["rank"]
    return clean, warnings


# ── 2. AIU level assignment (Question 1 only) ─────────────────────────────────

def validate_aiu_uniqueness(df_random_cutoff: pd.DataFrame, issues: list[dict]) -> bool:
    """
    Verify that each (dataset, keep_frac) maps to exactly one AIU value.

    AIU_level assignment assumes a 1:1 (dataset, keep_frac) -> AIU mapping; if
    a single keep_frac produced multiple distinct AIU values (e.g. due to a
    data/pipeline inconsistency), silently picking one would be wrong. This
    logs a 'multiple_aiu_per_keep_frac' issue (naming every offending AIU
    value) and returns False so the caller can abort the overall analysis
    instead of guessing.
    """
    grp = df_random_cutoff.groupby(["dataset", "keep_frac"])["AIU"].agg(lambda s: sorted(s.unique().tolist()))
    ok = True
    for (ds, kf), aiu_values in grp.items():
        if len(aiu_values) > 1:
            ok = False
            issues.append({
                "analysis": "overall", "ballot": f"overall/{ds}/keep_frac={kf}",
                "issue_type": "multiple_aiu_per_keep_frac",
                "detail": f"dataset={ds}, keep_frac={kf} has {len(aiu_values)} distinct AIU values: {aiu_values}",
            })
    return ok


def assign_aiu_levels(df_random_cutoff: pd.DataFrame) -> pd.DataFrame:
    """
    Within each dataset, order distinct (keep_frac, AIU) pairs by AIU ascending
    and assign ordinal AIU_L1 (sparsest) .. AIU_Ln (densest).

    Returns a small lookup frame: dataset, keep_frac, AIU, AIU_level.
    """
    pairs = (
        df_random_cutoff[["dataset", "keep_frac", "AIU"]]
        .drop_duplicates()
        .sort_values(["dataset", "AIU"])
        .reset_index(drop=True)
    )
    pairs["AIU_level"] = (
        pairs.groupby("dataset").cumcount() + 1
    ).map(lambda i: f"AIU_L{i}")
    return pairs


# ── 3. Question 1: overall ────────────────────────────────────────────────────

def run_overall(df: pd.DataFrame, issues: list[dict]) -> dict[str, dict[str, pd.DataFrame]]:
    base = df[(df["strategy"] == "random") & (df["cutoff"] == CUTOFF)].copy()
    if base.empty:
        issues.append({
            "analysis": "overall", "ballot": "", "issue_type": "no_data",
            "detail": "No rows found for strategy=random, cutoff=20.",
        })
        return {}

    if not validate_aiu_uniqueness(base, issues):
        print(
            "  [ABORT][overall] multiple_aiu_per_keep_frac detected -- "
            "overall analysis stopped; 01_full_borda_count.xlsx will contain "
            "only validation_issues (no data), 02/03 are not written."
        )
        empty = pd.DataFrame(columns=[
            "dataset", "keep_frac", "AIU", "AIU_level", "model", "ndcg_20", "rank", "borda_score"
        ])
        return {"01_full_borda_count": {"data": empty}}

    aiu_levels = assign_aiu_levels(base)
    base = base.merge(aiu_levels, on=["dataset", "keep_frac", "AIU"], how="left")

    full_rows = []
    for (ds, aiu_level), ballot in base.groupby(["dataset", "AIU_level"]):
        ranked, warns = rank_and_borda(ballot, score_col=METRIC_COL)
        ballot_key = f"overall/{ds}/{aiu_level}"
        for w in warns:
            issues.append({"analysis": "overall", "ballot": ballot_key, "issue_type": "duplicate_model", "detail": w})

        n_present = len(ranked)
        if n_present == 0:
            issues.append({
                "analysis": "overall", "ballot": ballot_key, "issue_type": "empty_ballot",
                "detail": "No model has a valid nDCG@20 value for this condition.",
            })
            continue
        if n_present != N_ALGORITHMS:
            issues.append({
                "analysis": "overall", "ballot": ballot_key, "issue_type": "incomplete_ballot",
                "detail": (
                    f"{n_present}/{N_ALGORITHMS} models present: {sorted(ranked['model'].tolist())} "
                    "-- entire ballot excluded from full_rows/downstream aggregation."
                ),
            })
            continue  # incomplete ballot: do NOT let any of its rows reach full_rows

        # n_present == N_ALGORITHMS here -- complete ballot, verify and keep.
        total = round(ranked["borda_score"].sum(), 6)
        if abs(total - EXPECTED_SUM) > 1e-6:
            issues.append({
                "analysis": "overall", "ballot": ballot_key, "issue_type": "borda_sum_mismatch",
                "detail": f"Expected total Borda {EXPECTED_SUM}, got {total}",
            })

        for _, r in ranked.iterrows():
            full_rows.append({
                "dataset":    ds,
                "keep_frac":  r["keep_frac"],
                "AIU":        r["AIU"],
                "AIU_level":  aiu_level,
                "model":      r["model"],
                "ndcg_20":    r[METRIC_COL],
                "rank":       r["rank"],
                "borda_score": r["borda_score"],
            })

    full_df = pd.DataFrame(full_rows)
    if full_df.empty:
        return {}
    full_df = full_df.sort_values(
        ["dataset", "AIU_level", "rank", "model"],
        key=lambda s: s if s.name != "AIU_level" else s.str.extract(r"(\d+)").astype(int)[0]
    ).reset_index(drop=True)

    # 02_borda_by_dataset: tidy (no averaging across AIU levels -- each level
    # already is a distinct experimental condition) + matrix per dataset.
    by_dataset_tidy = full_df[
        ["dataset", "AIU_level", "keep_frac", "AIU", "model", "borda_score", "rank"]
    ].sort_values(["dataset", "AIU_level", "rank"]).reset_index(drop=True)

    matrices = {}
    for ds in sorted(by_dataset_tidy["dataset"].unique()):
        mat = by_dataset_tidy[by_dataset_tidy["dataset"] == ds].pivot_table(
            index="model", columns="AIU_level", values="borda_score", aggfunc="first"
        )
        mat = mat.reindex(columns=sorted(mat.columns, key=lambda c: int(c.split("L")[1])))
        key = "HM_matrix" if ds == "hm" else "Amazon_matrix"
        matrices[key] = mat.reset_index()

    # 03_final_borda: average HM + Amazon per AIU_level x model.
    pivot = by_dataset_tidy.pivot_table(
        index=["AIU_level", "model"], columns="dataset", values="borda_score", aggfunc="first"
    ).reset_index()
    pivot = pivot.rename(columns={"hm": "HM_borda", "amazon": "Amazon_borda"})
    for c in ("HM_borda", "Amazon_borda"):
        if c not in pivot.columns:
            pivot[c] = np.nan
            issues.append({
                "analysis": "overall", "ballot": "final_borda", "issue_type": "dataset_missing_in_final",
                "detail": f"Column {c} absent when building final_borda -- filled with NaN.",
            })
    # final_borda is only valid when BOTH datasets have a Borda score for this
    # (AIU_level, model) -- skipna=False makes pandas.mean() return NaN rather
    # than silently averaging a single dataset when the other is missing.
    pivot["final_borda"] = pivot[["HM_borda", "Amazon_borda"]].mean(axis=1, skipna=False)

    one_missing = pivot["HM_borda"].isna() ^ pivot["Amazon_borda"].isna()
    for _, r in pivot[one_missing].iterrows():
        issues.append({
            "analysis": "overall",
            "ballot": f"final_borda/{r['AIU_level']}/{r['model']}",
            "issue_type": "missing_dataset_in_final_borda",
            "detail": (
                f"HM_borda={r['HM_borda']}, Amazon_borda={r['Amazon_borda']} "
                "-- final_borda set to NaN (not averaged from the single available dataset)."
            ),
        })

    final_rows = []
    for aiu_level, grp in pivot.groupby("AIU_level"):
        grp = grp.copy()
        grp["final_rank"] = np.nan
        valid = grp["final_borda"].notna()
        grp.loc[valid, "final_rank"] = grp.loc[valid, "final_borda"].rank(method="average", ascending=False)
        final_rows.append(grp)
    final_df = pd.concat(final_rows, ignore_index=True)
    final_df = final_df.sort_values(
        ["AIU_level", "final_rank"],
        key=lambda s: s if s.name != "AIU_level" else s.str.extract(r"(\d+)").astype(int)[0]
    ).reset_index(drop=True)
    final_df = final_df[["AIU_level", "model", "HM_borda", "Amazon_borda", "final_borda", "final_rank"]]

    final_matrix = final_df.pivot_table(index="model", columns="AIU_level", values="final_borda", aggfunc="first")
    final_matrix = final_matrix.reindex(columns=sorted(final_matrix.columns, key=lambda c: int(c.split("L")[1])))

    return {
        "01_full_borda_count": {"data": full_df},
        "02_borda_by_dataset": {"tidy": by_dataset_tidy, **matrices},
        "03_final_borda": {"data": final_df, "matrix": final_matrix.reset_index()},
    }


# ── 4/5. Question 2 & 3: group-based analyses (shared implementation) ────────

def run_group_analysis(
    df: pd.DataFrame,
    issues: list[dict],
    analysis_name: str,
    group_src_col: dict[str, str],
    group_order: list[str],
    group_out_col: str,  # "user_group" or "item_group"
) -> dict[str, dict[str, pd.DataFrame]]:
    base = df[(df["strategy"] == "random") & (df["cutoff"] == CUTOFF)].copy()
    if base.empty:
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "no_data",
            "detail": "No rows found for strategy=random, cutoff=20.",
        })
        return {}

    missing_cols = [c for c in group_src_col.values() if c not in base.columns]
    if missing_cols:
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "missing_group_columns",
            "detail": f"Missing source columns: {missing_cols}",
        })

    full_rows = []
    for group_label in group_order:
        src_col = group_src_col.get(group_label)
        if src_col is None or src_col not in base.columns:
            continue
        for (ds, kf), ballot in base.groupby(["dataset", "keep_frac"]):
            ballot_key = f"{analysis_name}/{ds}/keep_frac={kf}/{group_label}"
            ranked, warns = rank_and_borda(ballot, score_col=src_col)
            for w in warns:
                issues.append({"analysis": analysis_name, "ballot": ballot_key, "issue_type": "duplicate_model", "detail": w})

            n_present = len(ranked)
            if n_present == 0:
                issues.append({
                    "analysis": analysis_name, "ballot": ballot_key, "issue_type": "empty_ballot",
                    "detail": f"No model has a valid {src_col} value for this condition (likely empty cohort).",
                })
                continue
            if n_present != N_ALGORITHMS:
                issues.append({
                    "analysis": analysis_name, "ballot": ballot_key, "issue_type": "incomplete_ballot",
                    "detail": (
                        f"{n_present}/{N_ALGORITHMS} models present: {sorted(ranked['model'].tolist())} "
                        "-- entire ballot excluded from full_rows/downstream aggregation."
                    ),
                })
                continue  # incomplete ballot: do NOT let any of its rows reach full_rows

            # n_present == N_ALGORITHMS here -- complete ballot, verify and keep.
            total = round(ranked["borda_score"].sum(), 6)
            if abs(total - EXPECTED_SUM) > 1e-6:
                issues.append({
                    "analysis": analysis_name, "ballot": ballot_key, "issue_type": "borda_sum_mismatch",
                    "detail": f"Expected total Borda {EXPECTED_SUM}, got {total}",
                })

            for _, r in ranked.iterrows():
                full_rows.append({
                    "dataset":      ds,
                    "keep_frac":    kf,
                    "AIU":          r["AIU"],
                    group_out_col:  group_label,
                    "model":        r["model"],
                    "ndcg_20":      r[src_col],
                    "rank":         r["rank"],
                    "borda_score":  r["borda_score"],
                })

    full_df = pd.DataFrame(full_rows)
    if full_df.empty:
        return {}

    group_rank_key = {g: i for i, g in enumerate(group_order)}
    full_df["_go"] = full_df[group_out_col].map(group_rank_key)
    full_df = full_df.sort_values(["_go", "dataset", "keep_frac", "rank"]).drop(columns="_go").reset_index(drop=True)

    # 02_borda_by_dataset: average Borda across all valid keep_frac, per
    # dataset x group x model.
    agg = (
        full_df.groupby(["dataset", group_out_col, "model"])
        .agg(n_keep_frac=("keep_frac", "nunique"), avg_borda=("borda_score", "mean"))
        .reset_index()
    )
    agg["_go"] = agg[group_out_col].map(group_rank_key)
    agg = agg.sort_values(["dataset", "_go", "model"]).drop(columns="_go")

    rank_rows = []
    for (ds, grp_label), g in agg.groupby(["dataset", group_out_col]):
        g = g.copy()
        g["dataset_rank"] = g["avg_borda"].rank(method="average", ascending=False)
        rank_rows.append(g)
    by_dataset_tidy = pd.concat(rank_rows, ignore_index=True)
    by_dataset_tidy["_go"] = by_dataset_tidy[group_out_col].map(group_rank_key)
    by_dataset_tidy = (
        by_dataset_tidy.sort_values(["dataset", "_go", "dataset_rank"])
        .drop(columns="_go")
        .reset_index(drop=True)
    )
    by_dataset_tidy = by_dataset_tidy[
        ["dataset", group_out_col, "model", "n_keep_frac", "avg_borda", "dataset_rank"]
    ]

    matrices = {}
    for ds in sorted(by_dataset_tidy["dataset"].unique()):
        mat = by_dataset_tidy[by_dataset_tidy["dataset"] == ds].pivot_table(
            index="model", columns=group_out_col, values="avg_borda", aggfunc="first"
        )
        present_order = [g for g in group_order if g in mat.columns]
        mat = mat.reindex(columns=present_order)
        key = "HM_matrix" if ds == "hm" else "Amazon_matrix"
        matrices[key] = mat.reset_index()

    # 03_final_borda: average HM + Amazon avg_borda per group x model.
    pivot = by_dataset_tidy.pivot_table(
        index=[group_out_col, "model"], columns="dataset", values="avg_borda", aggfunc="first"
    ).reset_index()
    pivot = pivot.rename(columns={"hm": "HM_borda", "amazon": "Amazon_borda"})
    for c in ("HM_borda", "Amazon_borda"):
        if c not in pivot.columns:
            pivot[c] = np.nan
            issues.append({
                "analysis": analysis_name, "ballot": "final_borda", "issue_type": "dataset_missing_in_final",
                "detail": f"Column {c} absent when building final_borda -- filled with NaN.",
            })
    # final_borda is only valid when BOTH datasets have a Borda score for this
    # (group, model) -- skipna=False makes pandas.mean() return NaN rather
    # than silently averaging a single dataset when the other is missing.
    pivot["final_borda"] = pivot[["HM_borda", "Amazon_borda"]].mean(axis=1, skipna=False)

    one_missing = pivot["HM_borda"].isna() ^ pivot["Amazon_borda"].isna()
    for _, r in pivot[one_missing].iterrows():
        issues.append({
            "analysis": analysis_name,
            "ballot": f"final_borda/{r[group_out_col]}/{r['model']}",
            "issue_type": "missing_dataset_in_final_borda",
            "detail": (
                f"HM_borda={r['HM_borda']}, Amazon_borda={r['Amazon_borda']} "
                "-- final_borda set to NaN (not averaged from the single available dataset)."
            ),
        })

    final_rows = []
    for grp_label, g in pivot.groupby(group_out_col):
        g = g.copy()
        g["final_rank"] = np.nan
        valid = g["final_borda"].notna()
        g.loc[valid, "final_rank"] = g.loc[valid, "final_borda"].rank(method="average", ascending=False)
        final_rows.append(g)
    final_df = pd.concat(final_rows, ignore_index=True)
    final_df["_go"] = final_df[group_out_col].map(group_rank_key)
    final_df = final_df.sort_values(["_go", "final_rank"]).drop(columns="_go").reset_index(drop=True)
    final_df = final_df[[group_out_col, "model", "HM_borda", "Amazon_borda", "final_borda", "final_rank"]]

    final_matrix = final_df.pivot_table(index="model", columns=group_out_col, values="final_borda", aggfunc="first")
    present_order = [g for g in group_order if g in final_matrix.columns]
    final_matrix = final_matrix.reindex(columns=present_order)

    missing_groups = [g for g in group_order if g not in full_df[group_out_col].unique()]
    if missing_groups:
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "missing_expected_groups",
            "detail": f"Expected groups never produced any ballot data: {missing_groups}",
        })

    return {
        "01_full_borda_count": {"data": full_df},
        "02_borda_by_dataset": {"tidy": by_dataset_tidy, **matrices},
        "03_final_borda": {"data": final_df, "matrix": final_matrix.reset_index()},
    }


def run_coldstart_user(df: pd.DataFrame, issues: list[dict]) -> dict:
    return run_group_analysis(
        df, issues, "coldstart_user", USER_GROUP_SRC_COL, USER_GROUP_ORDER, "user_group"
    )


def run_longtail_item(df: pd.DataFrame, issues: list[dict]) -> dict:
    return run_group_analysis(
        df, issues, "longtail_item", ITEM_GROUP_SRC_COL, ITEM_GROUP_ORDER, "item_group"
    )


# ── 6. Collapsed-group analysis: Cold/NonCold, LongTail/Head ─────────────────
#
# Fixed thresholds (do not change here -- see the task spec):
#   Cold     = U1, U2_5                          (<=5 historical interactions)
#   NonCold  = U6_10, U11_20, U21_40, U41+        (>5 historical interactions)
#   LongTail = Pop1, Pop2_5, Pop6_10, Pop11_20    (<=20 historical interactions)
#   Head     = Pop21_40, Pop41+                   (>20 historical interactions)
#
# These are built DIRECTLY from the ballot-level full_df already produced by
# run_group_analysis (i.e. the exact 01_full_borda_count data) -- never from
# averaged nDCG, never by recomputing Borda, never from 03_final_borda. Every
# row in that full_df already belongs to a complete (8/8-model) ballot (see
# the "incomplete ballot" exclusion in run_group_analysis), so a plain
# groupby-mean over collapsed segments never mixes in a partial ballot.

# Rounded AIU-level display labels, specific to this project's chosen
# keep_frac values: both HM and Amazon were tuned (via preprocess.py
# KEEP_FRACS) so their 5 random-thinning levels land on roughly the same AIU
# milestones -- see overall/01_full_borda_count.xlsx for the exact per-dataset
# AIU values behind each level.
AIU_LEVEL_DISPLAY = {
    "AIU_L1": "<2", "AIU_L2": "4", "AIU_L3": "9", "AIU_L4": "15", "AIU_L5": ">20",
}

USER_COLLAPSE_MAP = {
    "U1": "Cold", "U2_5": "Cold",
    "U6_10": "NonCold", "U11_20": "NonCold", "U21_40": "NonCold", "U41+": "NonCold",
}
USER_SEGMENT_ORDER = ["Cold", "NonCold"]
USER_SEGMENT_DISPLAY = {"Cold": "Cold-start (≤5)", "NonCold": "Non-cold-start (>5)"}
USER_EXPECTED_BY_SEGMENT = {
    "Cold": ["U1", "U2_5"],
    "NonCold": ["U6_10", "U11_20", "U21_40", "U41+"],
}

ITEM_COLLAPSE_MAP = {
    "Pop1": "LongTail", "Pop2_5": "LongTail", "Pop6_10": "LongTail", "Pop11_20": "LongTail",
    "Pop21_40": "Head", "Pop41+": "Head",
}
ITEM_SEGMENT_ORDER = ["LongTail", "Head"]
ITEM_SEGMENT_DISPLAY = {"LongTail": "Long-tail (≤20)", "Head": "Head (>20)"}
ITEM_EXPECTED_BY_SEGMENT = {
    "LongTail": ["Pop1", "Pop2_5", "Pop6_10", "Pop11_20"],
    "Head": ["Pop21_40", "Pop41+"],
}


def validate_collapse_map(
    collapse_map: dict[str, str],
    expected_by_segment: dict[str, list[str]],
    analysis_name: str,
    issues: list[dict],
) -> bool:
    """Verify collapse_map maps exactly the expected detailed groups to each segment."""
    ok = True
    for segment, expected in expected_by_segment.items():
        actual = sorted(g for g, s in collapse_map.items() if s == segment)
        if actual != sorted(expected):
            ok = False
            issues.append({
                "analysis": analysis_name, "ballot": f"collapse_map/{segment}",
                "issue_type": "collapse_map_mismatch",
                "detail": f"Expected {segment} = {sorted(expected)}, got {actual}",
            })
    return ok


def collapse_and_aggregate(
    full_df: pd.DataFrame,
    collapse_map: dict[str, str],
    segment_order: list[str],
    detailed_group_col: str,   # "user_group" or "item_group"
    segment_col: str,          # "user_segment" or "item_segment"
    analysis_name: str,
    issues: list[dict],
    aiu_level_lookup: pd.DataFrame,
) -> dict[str, pd.DataFrame] | None:
    """
    Collapse the detailed-group ballot-level Borda scores (from 01_full_borda_count)
    into 2-way segments (Cold/NonCold or LongTail/Head), average per
    dataset x segment x model, rank within each dataset, then combine HM+Amazon
    into a final Borda/rank per segment x model (NaN if either dataset missing).
    """
    if full_df is None or full_df.empty:
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "no_data",
            "detail": "01_full_borda_count data is empty -- cannot build collapsed segments.",
        })
        return None

    work = full_df.copy()
    work[segment_col] = work[detailed_group_col].map(collapse_map)
    unmapped = sorted(work[work[segment_col].isna()][detailed_group_col].dropna().unique().tolist())
    if unmapped:
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "unmapped_detailed_group",
            "detail": f"Detailed group(s) not covered by the collapse map, excluded from segments: {unmapped}",
        })
        work = work.dropna(subset=[segment_col])

    # Per-(dataset, keep_frac/AIU, segment, model) breakdown -- unlike the
    # dataset-level aggregation below, this keeps keep_frac/AIU as a row
    # dimension instead of averaging it away, by averaging only the detailed
    # groups that fall within the same segment AT THE SAME keep_frac (1 or 2
    # of them, whichever are non-empty there).
    by_keep_frac = (
        work.groupby(["dataset", "keep_frac", "AIU", segment_col, "model"])
        .agg(n_detailed_groups=("borda_score", "size"), avg_borda=("borda_score", "mean"))
        .reset_index()
    )
    # AIU_level comes from the SAME lookup as the "overall" (Q1) analysis
    # (assign_aiu_levels on the full strategy=random/cutoff=20 data) -- not
    # recomputed here -- so "AIU_L3" means the exact same ordinal sparsity
    # level in both places, and HM/Amazon line up at matching levels even
    # though their actual keep_frac values differ.
    by_keep_frac = by_keep_frac.merge(
        aiu_level_lookup[["dataset", "keep_frac", "AIU_level"]],
        on=["dataset", "keep_frac"], how="left",
    )
    unmatched_level = by_keep_frac[by_keep_frac["AIU_level"].isna()]
    if not unmatched_level.empty:
        bad = sorted(unmatched_level[["dataset", "keep_frac"]].drop_duplicates().apply(tuple, axis=1).tolist())
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "aiu_level_lookup_miss",
            "detail": f"(dataset, keep_frac) pairs not found in the overall AIU_level lookup: {bad}",
        })

    # Each ballot (dataset, keep_frac, detailed_group) contributes exactly one
    # row per model since full_df only ever holds complete (8/8) ballots --
    # so a plain row count per (dataset, segment, model) already equals the
    # number of distinct detailed-group ballots contributing to that model.
    agg = (
        work.groupby(["dataset", segment_col, "model"])
        .agg(n_ballots=("borda_score", "size"), avg_borda=("borda_score", "mean"))
        .reset_index()
    )

    seg_rank_key = {s: i for i, s in enumerate(segment_order)}

    rank_rows = []
    for (ds, seg), g in agg.groupby(["dataset", segment_col]):
        g = g.copy()
        if len(g) != N_ALGORITHMS:
            issues.append({
                "analysis": analysis_name, "ballot": f"{ds}/{seg}",
                "issue_type": "incomplete_collapsed_group",
                "detail": (
                    f"{len(g)}/{N_ALGORITHMS} models have contributing ballots for "
                    f"dataset={ds}, segment={seg}: {sorted(g['model'].tolist())} -- "
                    "excluded from dataset_rank / by_dataset output."
                ),
            })
            continue
        g["dataset_rank"] = g["avg_borda"].rank(method="average", ascending=False)
        rank_rows.append(g)

    if not rank_rows:
        issues.append({
            "analysis": analysis_name, "ballot": "", "issue_type": "no_data",
            "detail": "No dataset x segment group had all 8 models -- collapsed analysis produced no output.",
        })
        return None

    by_dataset_tidy = pd.concat(rank_rows, ignore_index=True)
    by_dataset_tidy["_so"] = by_dataset_tidy[segment_col].map(seg_rank_key)
    by_dataset_tidy = (
        by_dataset_tidy.sort_values(["dataset", "_so", "dataset_rank"])
        .drop(columns="_so")
        .reset_index(drop=True)
    )
    by_dataset_tidy = by_dataset_tidy[
        ["dataset", segment_col, "model", "n_ballots", "avg_borda", "dataset_rank"]
    ]

    matrices = {}
    for ds in sorted(by_dataset_tidy["dataset"].unique()):
        mat = by_dataset_tidy[by_dataset_tidy["dataset"] == ds].pivot_table(
            index="model", columns=segment_col, values="avg_borda", aggfunc="first"
        )
        present_order = [s for s in segment_order if s in mat.columns]
        mat = mat.reindex(columns=present_order)
        key = "HM_matrix" if ds == "hm" else "Amazon_matrix"
        matrices[key] = mat.reset_index()

    # Final: average HM + Amazon avg_borda per segment x model -- NaN if either
    # dataset is missing (never silently averaged from a single dataset).
    pivot = by_dataset_tidy.pivot_table(
        index=[segment_col, "model"], columns="dataset", values="avg_borda", aggfunc="first"
    ).reset_index()
    pivot = pivot.rename(columns={"hm": "HM_borda", "amazon": "Amazon_borda"})
    for c in ("HM_borda", "Amazon_borda"):
        if c not in pivot.columns:
            pivot[c] = np.nan
            issues.append({
                "analysis": analysis_name, "ballot": "final_borda", "issue_type": "dataset_missing_in_final",
                "detail": f"Column {c} absent when building collapsed final_borda -- filled with NaN.",
            })
    pivot["final_borda"] = pivot[["HM_borda", "Amazon_borda"]].mean(axis=1, skipna=False)

    one_missing = pivot["HM_borda"].isna() ^ pivot["Amazon_borda"].isna()
    for _, r in pivot[one_missing].iterrows():
        issues.append({
            "analysis": analysis_name,
            "ballot": f"final_borda/{r[segment_col]}/{r['model']}",
            "issue_type": "missing_dataset_in_final_borda",
            "detail": (
                f"HM_borda={r['HM_borda']}, Amazon_borda={r['Amazon_borda']} "
                "-- final_borda set to NaN (not averaged from the single available dataset)."
            ),
        })

    final_rows = []
    for seg, g in pivot.groupby(segment_col):
        g = g.copy()
        g["final_rank"] = np.nan
        valid = g["final_borda"].notna()
        g.loc[valid, "final_rank"] = g.loc[valid, "final_borda"].rank(method="average", ascending=False)
        final_rows.append(g)
    final_df = pd.concat(final_rows, ignore_index=True)
    final_df["_so"] = final_df[segment_col].map(seg_rank_key)
    final_df = final_df.sort_values(["_so", "final_rank"]).drop(columns="_so").reset_index(drop=True)
    final_df = final_df[[segment_col, "model", "HM_borda", "Amazon_borda", "final_borda", "final_rank"]]

    final_matrix = final_df.pivot_table(index="model", columns=segment_col, values="final_borda", aggfunc="first")
    present_order = [s for s in segment_order if s in final_matrix.columns]
    final_matrix = final_matrix.reindex(columns=present_order)

    by_keep_frac["_so"] = by_keep_frac[segment_col].map(seg_rank_key)
    by_keep_frac = (
        by_keep_frac.sort_values(["dataset", "_so", "keep_frac", "model"])
        .drop(columns="_so")
        .reset_index(drop=True)
    )
    by_keep_frac = by_keep_frac[
        ["dataset", "keep_frac", "AIU", "AIU_level", segment_col, "model", "n_detailed_groups", "avg_borda"]
    ]

    # Combined HM+Amazon per AIU_level x segment x model -- datasets are
    # matched by ordinal AIU_level (same sparsity rank), not by raw keep_frac,
    # since HM and Amazon use different keep_frac values to reach comparable
    # AIU. NaN final_borda if either dataset is missing at that level (never
    # silently averaged from one dataset alone).
    by_aiu_pivot = by_keep_frac.pivot_table(
        index=["AIU_level", segment_col, "model"], columns="dataset", values="avg_borda", aggfunc="first"
    ).reset_index()
    by_aiu_pivot = by_aiu_pivot.rename(columns={"hm": "HM_borda", "amazon": "Amazon_borda"})
    for c in ("HM_borda", "Amazon_borda"):
        if c not in by_aiu_pivot.columns:
            by_aiu_pivot[c] = np.nan
    by_aiu_pivot["final_borda"] = by_aiu_pivot[["HM_borda", "Amazon_borda"]].mean(axis=1, skipna=False)

    one_missing_aiu = by_aiu_pivot["HM_borda"].isna() ^ by_aiu_pivot["Amazon_borda"].isna()
    for _, r in by_aiu_pivot[one_missing_aiu].iterrows():
        issues.append({
            "analysis": analysis_name,
            "ballot": f"final_by_aiu_level/{r['AIU_level']}/{r[segment_col]}/{r['model']}",
            "issue_type": "missing_dataset_in_final_borda",
            "detail": (
                f"HM_borda={r['HM_borda']}, Amazon_borda={r['Amazon_borda']} "
                "-- final_borda set to NaN (not averaged from the single available dataset)."
            ),
        })

    final_by_aiu_rows = []
    for (lvl, seg), g in by_aiu_pivot.groupby(["AIU_level", segment_col]):
        g = g.copy()
        g["final_rank"] = np.nan
        valid = g["final_borda"].notna()
        g.loc[valid, "final_rank"] = g.loc[valid, "final_borda"].rank(method="average", ascending=False)
        final_by_aiu_rows.append(g)
    final_by_aiu = pd.concat(final_by_aiu_rows, ignore_index=True)
    final_by_aiu["AIU_level_display"] = final_by_aiu["AIU_level"].map(AIU_LEVEL_DISPLAY).fillna(final_by_aiu["AIU_level"])
    final_by_aiu["_lo"] = final_by_aiu["AIU_level"].str.extract(r"(\d+)").astype(int)
    final_by_aiu["_so"] = final_by_aiu[segment_col].map(seg_rank_key)
    final_by_aiu = (
        final_by_aiu.sort_values(["_so", "_lo", "final_rank"])
        .drop(columns=["_so", "_lo"])
        .reset_index(drop=True)
    )
    final_by_aiu = final_by_aiu[
        ["AIU_level", "AIU_level_display", segment_col, "model", "HM_borda", "Amazon_borda", "final_borda", "final_rank"]
    ]

    return {
        "by_dataset_tidy": by_dataset_tidy,
        "by_keep_frac": by_keep_frac,
        "final_by_aiu_level": final_by_aiu,
        "matrices": matrices,
        "final_data": final_df,
        "final_matrix": final_matrix.reset_index(),
    }


def run_coldstart_segments(
    full_df: pd.DataFrame, issues: list[dict], aiu_level_lookup: pd.DataFrame,
) -> dict[str, pd.DataFrame] | None:
    validate_collapse_map(USER_COLLAPSE_MAP, USER_EXPECTED_BY_SEGMENT, "coldstart_user_segments", issues)
    return collapse_and_aggregate(
        full_df, USER_COLLAPSE_MAP, USER_SEGMENT_ORDER,
        "user_group", "user_segment", "coldstart_user_segments", issues, aiu_level_lookup,
    )


def run_longtail_segments(
    full_df: pd.DataFrame, issues: list[dict], aiu_level_lookup: pd.DataFrame,
) -> dict[str, pd.DataFrame] | None:
    validate_collapse_map(ITEM_COLLAPSE_MAP, ITEM_EXPECTED_BY_SEGMENT, "longtail_item_segments", issues)
    return collapse_and_aggregate(
        full_df, ITEM_COLLAPSE_MAP, ITEM_SEGMENT_ORDER,
        "item_group", "item_segment", "longtail_item_segments", issues, aiu_level_lookup,
    )


# ── 7. Segment Borda bar charts ───────────────────────────────────────────────

# Reuses the 8-color categorical palette already established in
# scripts/plot_performance_charts.py (_BASE_COLORS) for visual consistency
# across the project's chart outputs, without importing that module (it has
# a different input/pipeline and no "grouped segment bar" chart type).
_CHART_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


def plot_segment_borda_chart(
    final_df: pd.DataFrame,
    segment_col: str,
    segment_order: list[str],
    segment_display: dict[str, str],
    title: str,
    out_path: Path,
) -> dict:
    """Grouped bar chart: x=8 algorithms, one bar pair per segment, y=final_borda."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted(final_df["model"].unique().tolist())
    segments = [s for s in segment_order if s in final_df[segment_col].unique()]
    if not models or not segments:
        return {"status": "skip", "reason": "no models/segments to plot"}

    pivot = final_df.pivot_table(index="model", columns=segment_col, values="final_borda", aggfunc="first")
    pivot = pivot.reindex(index=models, columns=segments)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(models))
    bar_w = 0.8 / max(len(segments), 1)

    for i, seg in enumerate(segments):
        vals = pivot[seg].to_numpy(dtype=float)
        offset = (i - (len(segments) - 1) / 2) * bar_w
        color = _CHART_COLORS[i % len(_CHART_COLORS)]
        bars = ax.bar(x + offset, np.nan_to_num(vals, nan=0.0), width=bar_w,
                      label=segment_display.get(seg, seg), color=color)
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Final Borda Score", fontsize=10)
    ax.set_ylim(0, 7)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"status": "ok", "output_path": str(out_path)}


# ── Excel export ──────────────────────────────────────────────────────────────

def _autofit(ws, max_width=40):
    for col_cells in ws.columns:
        max_len = max((len(str(c.value)) if c.value is not None else 0) for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, max_width)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def _apply_number_formats(ws, df: pd.DataFrame):
    ndcg_fmt  = "0.000000"
    borda_fmt = "0.000"
    for col_idx, col_name in enumerate(df.columns, start=1):
        lname = str(col_name).lower()
        if "ndcg" in lname:
            fmt = ndcg_fmt
        elif "borda" in lname:
            fmt = borda_fmt
        elif lname in ("rank", "final_rank", "dataset_rank", "aiu"):
            fmt = "0.00"
        else:
            fmt = None
        if fmt:
            for row in ws.iter_rows(min_row=2, min_col=col_idx, max_col=col_idx):
                for cell in row:
                    cell.number_format = fmt


def write_workbook(sheets: dict[str, pd.DataFrame], validation_issues: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            if frame is None:
                continue
            safe_name = name[:31]
            frame.to_excel(writer, sheet_name=safe_name, index=False)
            ws = writer.sheets[safe_name]
            _autofit(ws)
            _apply_number_formats(ws, frame)
        if not validation_issues.empty:
            validation_issues.to_excel(writer, sheet_name="validation_issues", index=False)
            ws = writer.sheets["validation_issues"]
            _autofit(ws)
    print(f"  Written: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Wiring for the collapsed-segment layer (section 6/5 of the spec): which
# runner/columns/filenames/chart apply to which of the two group analyses.
SEGMENT_CONFIG = {
    "coldstart_user": {
        "runner": run_coldstart_segments,
        "segment_col": "user_segment",
        "segment_order": USER_SEGMENT_ORDER,
        "segment_display": USER_SEGMENT_DISPLAY,
        "file04": "04_borda_by_dataset_cold_vs_uncold",
        "file05": "05_final_borda_cold_vs_uncold",
        "chart_name": "cold_vs_noncold_borda.png",
        "chart_title": "Algorithm Ranking for Cold-start vs Non-cold-start Users",
    },
    "longtail_item": {
        "runner": run_longtail_segments,
        "segment_col": "item_segment",
        "segment_order": ITEM_SEGMENT_ORDER,
        "segment_display": ITEM_SEGMENT_DISPLAY,
        "file04": "04_borda_by_dataset_long_vs_unlong",
        "file05": "05_final_borda_long_vs_unlong",
        "chart_name": "longtail_vs_head_borda.png",
        "chart_title": "Algorithm Ranking for Long-tail vs Head Items",
    },
}


def run_segment_layer(
    analysis_name: str, sheets_by_file: dict, out_dir: Path, report: dict,
    aiu_level_lookup: pd.DataFrame,
) -> None:
    """Build the 04/05 collapsed-segment Excel files + chart for one analysis
    (coldstart_user or longtail_item), from the 01_full_borda_count data that
    was just computed -- does not touch/recompute 01, 02 or 03."""
    cfg = SEGMENT_CONFIG[analysis_name]
    full_df = sheets_by_file.get("01_full_borda_count", {}).get("data")

    seg_issues: list[dict] = []
    seg_result = cfg["runner"](full_df, seg_issues, aiu_level_lookup)
    seg_issues_df = pd.DataFrame(seg_issues) if seg_issues else pd.DataFrame(
        columns=["analysis", "ballot", "issue_type", "detail"])

    print(f"\n  --- {analysis_name} segments ({'/'.join(cfg['segment_order'])}) ---")
    if seg_result is None:
        print("  [WARN] segment analysis produced no output:")
        for _, r in seg_issues_df.iterrows():
            print(f"    [{r['issue_type']}] {r['ballot']}: {r['detail']}")
        report[f"{analysis_name}_segments"] = {"issues": seg_issues_df}
        return

    file04_sheets = {
        "tidy": seg_result["by_dataset_tidy"],
        "by_keep_frac": seg_result["by_keep_frac"],
        **seg_result["matrices"],
    }
    file05_sheets = {
        "data": seg_result["final_data"],
        "matrix": seg_result["final_matrix"],
        "by_aiu_level": seg_result["final_by_aiu_level"],
    }

    empty_issues = pd.DataFrame(columns=["analysis", "ballot", "issue_type", "detail"])
    write_workbook(file04_sheets, empty_issues, out_dir / f"{cfg['file04']}.xlsx")
    write_workbook(file05_sheets, seg_issues_df, out_dir / f"{cfg['file05']}.xlsx")

    chart_path = out_dir / "charts" / cfg["chart_name"]
    chart_result = plot_segment_borda_chart(
        seg_result["final_data"], cfg["segment_col"], cfg["segment_order"],
        cfg["segment_display"], cfg["chart_title"], chart_path,
    )
    print(f"  Chart: {chart_result}")

    print(f"  segment validation issues: {len(seg_issues_df)}")
    for _, r in seg_issues_df.iterrows():
        print(f"    [{r['issue_type']}] {r['ballot']}: {r['detail']}")

    report[f"{analysis_name}_segments"] = {
        "by_dataset_tidy": seg_result["by_dataset_tidy"],
        "final_data": seg_result["final_data"],
        "chart": chart_result,
        "issues": seg_issues_df,
    }


def main():
    print(f"Loading: {INPUT_XLSX}")
    df = load_long_all_results(INPUT_XLSX)
    print(f"  {len(df)} rows loaded from long_all_results")

    global_issues: list[dict] = []
    models = validate_input_assumptions(df, global_issues)
    print(f"  Models found ({len(models)}): {models}")
    for issue in global_issues:
        print(f"  [VALIDATION][{issue['analysis']}] {issue['issue_type']}: {issue['detail']}")

    report: dict[str, dict] = {}

    # Shared AIU_level lookup (dataset, keep_frac, AIU -> AIU_L1..AIU_Ln),
    # computed once from the same strategy=random/cutoff=20 data run_overall()
    # uses -- so the segment-layer "by_aiu_level" sheets below line up with
    # Q1's AIU_level exactly, rather than risking a second, possibly-divergent
    # ordinal assignment.
    random_cutoff_df = df[(df["strategy"] == "random") & (df["cutoff"] == CUTOFF)]
    validate_aiu_uniqueness(random_cutoff_df, global_issues)  # same guard run_overall() uses
    aiu_level_lookup = assign_aiu_levels(random_cutoff_df)

    for analysis_name, runner in (
        ("overall", run_overall),
        ("coldstart_user", run_coldstart_user),
        ("longtail_item", run_longtail_item),
    ):
        print(f"\n=== {analysis_name} ===")
        issues: list[dict] = list(global_issues)  # carry global issues into every workbook too
        sheets_by_file = runner(df, issues)
        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame(
            columns=["analysis", "ballot", "issue_type", "detail"])

        out_dir = OUTPUT_ROOT / analysis_name
        file_titles = {
            "01_full_borda_count": "data",
            "02_borda_by_dataset": "borda_by_dataset",
            "03_final_borda": "final_borda",
        }
        n_ballots = 0
        n_rows_full = 0
        for file_stem, sheets in sheets_by_file.items():
            out_path = out_dir / f"{file_stem}.xlsx"
            # only attach validation_issues to the 01 file to avoid tripling it
            vdf = issues_df if file_stem == "01_full_borda_count" else pd.DataFrame(
                columns=["analysis", "ballot", "issue_type", "detail"])
            write_workbook(sheets, vdf, out_path)
            if file_stem == "01_full_borda_count" and "data" in sheets:
                n_rows_full = len(sheets["data"])
                ballot_cols = [c for c in ("dataset", "AIU_level", "user_group", "item_group", "keep_frac") if c in sheets["data"].columns]
                n_ballots = sheets["data"].drop_duplicates(subset=ballot_cols).shape[0] if ballot_cols else 0

        report[analysis_name] = {
            "n_rows_full": n_rows_full,
            "n_issues": len(issues_df),
            "issues": issues_df,
        }
        print(f"  full_borda_count rows: {n_rows_full}")
        print(f"  validation issues: {len(issues_df)}")
        if not issues_df.empty:
            for _, r in issues_df.head(20).iterrows():
                print(f"    [{r['issue_type']}] {r['ballot']}: {r['detail']}")
            if len(issues_df) > 20:
                print(f"    ... and {len(issues_df) - 20} more (see validation_issues sheet)")

        if analysis_name in SEGMENT_CONFIG:
            run_segment_layer(analysis_name, sheets_by_file, out_dir, report, aiu_level_lookup)

    print("\nDone.")
    return report


if __name__ == "__main__":
    main()
