"""
Build unified performance summary Excel from Elliot and RecBole result files.

Usage:
    python scripts/build_performance_summary.py
    python scripts/build_performance_summary.py --output results/_summary/performance_summary.xlsx
    python scripts/build_performance_summary.py --expected-models VSM FunkSVD ItemKNN BPR
    python scripts/build_performance_summary.py --model-frameworks "ItemKNN:elliot,recbole"
    python scripts/build_performance_summary.py --allow-audit-errors
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ELLIOT_ROOT  = PROJECT_ROOT / "results" / "elliot"
DEFAULT_RECBOLE_ROOT = PROJECT_ROOT / "results" / "recbole"
DEFAULT_PROCESSED    = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS      = PROJECT_ROOT / "data" / "reports"
DEFAULT_OUTPUT       = PROJECT_ROOT / "results" / "_summary" / "performance_summary.xlsx"

# ── Constants ─────────────────────────────────────────────────────────────────

MAIN_METRICS     = ["Precision", "Recall", "nDCG", "MAP", "MRR"]
COLD_U1_METRICS  = ["Recall_u1", "nDCG_u1"]
# Canonical beyond-accuracy column names used in long_df and summary sheets.
# Elliot native → canonical:  ARP→AveragePopularity, APLT→TailPercentage (rename at collect time)
# RecBole source: *_evaluation.tsv written by external/recbole/run_evaluation.py (already canonical)
BEYOND_ACCURACY_METRICS = ["ItemCoverage", "AveragePopularity", "Gini", "TailPercentage"]
POP_GROUP_SUFFIXES  = ["pop1", "pop2_5", "pop6_10", "pop11_20", "pop20plus"]
USER_GROUP_SUFFIXES = ["u1", "u2_5", "u6_10", "u11_20", "u20plus"]
POP_METRICS  = [f"Recall_{s}" for s in POP_GROUP_SUFFIXES] + [f"nDCG_{s}" for s in POP_GROUP_SUFFIXES]
USER_METRICS = [f"Recall_{s}" for s in USER_GROUP_SUFFIXES] + [f"nDCG_{s}" for s in USER_GROUP_SUFFIXES]
# Full metric set carried through build_long / check_and_dedup / build_wide
RESULT_METRICS = MAIN_METRICS + BEYOND_ACCURACY_METRICS + POP_METRICS + USER_METRICS

# Mapping: canonical name → Elliot TSV column name (Elliot uses its own metric class names)
_ELLIOT_BEYOND_COL: dict[str, str] = {
    "ItemCoverage":      "ItemCoverage",  # same in both
    "AveragePopularity": "ARP",           # Elliot class ARP
    "Gini":              "Gini",          # same in both
    "TailPercentage":    "APLT",          # Elliot class APLT
}

# Mapping: RecBole evaluation.tsv raw metric name (lowercase, no @cutoff) → canonical
# Used when parsing *_evaluation.tsv wide-format files
_RECBOLE_EVAL_COL: dict[str, str] = {
    "precision": "Precision", "recall": "Recall", "ndcg": "nDCG",
    "map": "MAP", "mrr": "MRR",
    "itemcoverage": "ItemCoverage", "averagepopularity": "AveragePopularity",
    "giniindex": "Gini", "tailpercentage": "TailPercentage",
    **{f"recall_{s}": f"Recall_{s}" for s in POP_GROUP_SUFFIXES},
    **{f"ndcg_{s}":   f"nDCG_{s}"   for s in POP_GROUP_SUFFIXES},
    **{f"recall_{s}": f"Recall_{s}" for s in USER_GROUP_SUFFIXES},
    **{f"ndcg_{s}":   f"nDCG_{s}"   for s in USER_GROUP_SUFFIXES},
}

EXPECTED_MODELS  = ["VSM", "FunkSVD", "ItemKNN", "BPR"]
EXPECTED_CUTOFFS = [10, 20, 50]
VALID_STRATEGIES = {"base", "random", "head", "tail"}

_STRATEGY_ORDER  = {"base": 0, "head": 1, "random": 2, "tail": 3}

SPARSITY_OUT = [
    "n_interactions", "n_interactions_x1000",
    "oss_pct", "uss", "iss",
    "user_gini", "item_gini",
    "coldstart_user_pct", "coldstart_item_pct",
]

# Default expected framework(s) per model — used ONLY when the model is not
# detected in any framework from actual data AND no CLI override is given.
DEFAULT_MODEL_FRAMEWORKS: dict[str, list[str]] = {
    "VSM":     ["elliot"],
    "FunkSVD": ["elliot"],
    "ItemKNN": ["elliot"],
    "BPR":     ["recbole"],
}

# Short canonical name lookup (prefix of full model string, lowercased)
_MODEL_NORM: dict[str, str] = {
    "funksvd":  "FunkSVD",
    "vsm":      "VSM",
    "itemknn":  "ItemKNN",
    "bpr":      "BPR",
    "neumf":    "NeuMF",
    "mostpop":  "MostPop",
    "fm":       "FM",
    "lightgcn": "LightGCN",
    "ngcf":     "NGCF",
}

# ── Experiment name patterns (Fix #7: support underscores/digits in dataset) ──

_EXP_KEEP_PAT = re.compile(
    r"^(?P<dataset>.+?)_(?P<strategy>random|head|tail)_keep(?P<keep_frac>\d+(?:\.\d+)?)$"
)
_EXP_BASE_PAT = re.compile(r"^(?P<dataset>.+?)_dedup_base_split$")


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalize_model(raw: str) -> str | None:
    """'FunkSVD_seed=42_...' → 'FunkSVD'. Returns None if unrecognised."""
    prefix = str(raw).split("_")[0].lower()
    return _MODEL_NORM.get(prefix)


def parse_experiment_name(name: str) -> dict:
    """
    Parse experiment folder name into dataset / strategy / keep_frac.
    Supports dataset names with underscores, digits, hyphens.
    """
    m = _EXP_KEEP_PAT.match(name)
    if m:
        kf = float(m.group("keep_frac"))
        if not (0 < kf <= 1):
            raise ValueError(f"keep_frac={kf} not in (0,1] for experiment '{name}'")
        strat = m.group("strategy")
        return {"dataset": m.group("dataset"), "strategy": strat, "keep_frac": kf}

    m = _EXP_BASE_PAT.match(name)
    if m:
        return {"dataset": m.group("dataset"), "strategy": "base", "keep_frac": 1.0}

    raise ValueError(f"Cannot parse experiment name: '{name}'")


def load_metadata(exp_name: str, processed_root: Path) -> dict:
    """Load k_user, k_item and train_metrics from metadata.json."""
    mj = processed_root / exp_name / "metadata.json"
    if not mj.exists():
        return {}
    try:
        data = json.loads(mj.read_text(encoding="utf-8"))
        result = {
            "k_user": data.get("k_user"),
            "k_item": data.get("k_item"),
        }
        tm = data.get("train_metrics", {})
        if tm:
            ni = tm.get("n_interactions", float("nan"))
            result.update({
                "n_interactions":       ni,
                "n_interactions_x1000": ni / 1000,
                "oss_pct":              tm.get("oss", float("nan")) * 100,
                "uss":                  tm.get("uss", float("nan")),
                "iss":                  tm.get("iss", float("nan")),
                "user_gini":            tm.get("user_gini", float("nan")),
                "item_gini":            tm.get("item_gini", float("nan")),
                "coldstart_user_pct":   tm.get("coldstart_user", float("nan")) * 100,
                "coldstart_item_pct":   tm.get("coldstart_item", float("nan")) * 100,
            })
        return result
    except Exception as e:
        print(f"[WARN] metadata.json read error for {exp_name}: {e}")
        return {}


# ── Collectors ────────────────────────────────────────────────────────────────

def collect_elliot(elliot_root: Path, processed_root: Path, audit: list) -> list[dict]:
    """Collect rows from results/elliot/{exp}/performance/{Model}_cutoff_{N}.tsv."""
    if not elliot_root.exists():
        print(f"[WARN] Elliot results root not found: {elliot_root}")
        return []

    rows = []
    for exp_dir in sorted(elliot_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        perf_dir = exp_dir / "performance"
        if not perf_dir.exists():
            continue

        try:
            exp_meta = parse_experiment_name(exp_dir.name)
        except ValueError as e:
            audit.append({"level": "ERROR", "source": str(exp_dir), "message": str(e)})
            continue

        metadata = load_metadata(exp_dir.name, processed_root)

        for tsv in sorted(perf_dir.glob("*_cutoff_*.tsv")):
            m = re.match(r"^(.+)_cutoff_(\d+)\.tsv$", tsv.name)
            if not m:
                continue
            fname_model = m.group(1)
            cutoff = int(m.group(2))

            if cutoff <= 0:
                audit.append({"level": "ERROR", "source": str(tsv),
                               "message": f"cutoff={cutoff} is not a positive integer"})
                continue

            try:
                df = pd.read_csv(tsv, sep="\t")
            except Exception as e:
                audit.append({"level": "ERROR", "source": str(tsv), "message": f"Read error: {e}"})
                continue

            if df.empty or "model" not in df.columns:
                audit.append({"level": "WARN", "source": str(tsv), "message": "Empty or no 'model' column"})
                continue

            for _, row in df.iterrows():
                full_model = str(row.get("model", ""))
                content_model = normalize_model(full_model)
                fname_model_norm = normalize_model(fname_model)

                if content_model is None:
                    audit.append({"level": "WARN", "source": str(tsv),
                                   "message": f"Unrecognised model in content: '{full_model}'"})
                    content_model = fname_model

                if fname_model_norm and content_model != fname_model_norm:
                    audit.append({
                        "level": "WARN", "source": str(tsv),
                        "message": (f"Model name mismatch: filename says '{fname_model_norm}', "
                                    f"content says '{content_model}' (using content)")
                    })

                record = {
                    "framework":   "elliot",
                    "experiment":  exp_dir.name,
                    "source_file": str(tsv),
                    "model":       content_model,
                    "full_model":  full_model,
                    "cutoff":      cutoff,
                    **exp_meta,
                    **{k: metadata.get(k) for k in ["k_user", "k_item"]},
                    **{k: metadata.get(k, float("nan")) for k in SPARSITY_OUT if k not in ["k_user", "k_item"]},
                }
                for metric in MAIN_METRICS + POP_METRICS + USER_METRICS:
                    v = row.get(metric, float("nan"))
                    record[metric] = float(v) if pd.notna(v) else float("nan")
                    if pd.notna(v) and not (0 <= float(v) <= 1):
                        audit.append({
                            "level": "WARN", "source": str(tsv),
                            "message": f"metric {metric}={v} out of [0,1]"
                        })
                for canonical, elliot_col in _ELLIOT_BEYOND_COL.items():
                    v = row.get(elliot_col, float("nan"))
                    record[canonical] = float(v) if pd.notna(v) else float("nan")
                rows.append(record)

    return rows


def _safe_cutoff(raw_val, source: str, audit: list) -> int | None:
    """
    Safely convert a cutoff cell to int. (Fix #8)
    Rejects NaN, non-integer floats (e.g. 20.5), and non-positive values.
    Returns None on failure (caller should skip the row).
    """
    numeric = pd.to_numeric(raw_val, errors="coerce")
    if pd.isna(numeric):
        audit.append({"level": "ERROR", "source": source,
                       "message": f"cutoff='{raw_val}' is not numeric"})
        return None
    if not float(numeric).is_integer():
        audit.append({"level": "ERROR", "source": source,
                       "message": f"cutoff={numeric} is not an integer (non-integer float rejected)"})
        return None
    cutoff = int(numeric)
    if cutoff <= 0:
        audit.append({"level": "ERROR", "source": source,
                       "message": f"cutoff={cutoff} must be a positive integer"})
        return None
    return cutoff


def _load_beyond_index(perf_dir: Path) -> dict[tuple, dict]:
    """Load all *_beyond.tsv files in perf_dir; return {(model_norm, cutoff): {metric: value}}."""
    idx: dict[tuple, dict] = {}
    for bf in sorted(perf_dir.glob("*_beyond.tsv")):
        try:
            bdf = pd.read_csv(bf, sep="\t")
        except Exception:
            continue
        if bdf.empty or "model" not in bdf.columns or "cutoff" not in bdf.columns:
            continue
        for _, br in bdf.iterrows():
            mn  = normalize_model(str(br.get("model", ""))) or str(br.get("model", ""))
            cut = _safe_cutoff(br["cutoff"], str(bf), [])
            if cut is None:
                continue
            key = (mn, cut)
            idx[key] = {m: float(br[m]) for m in BEYOND_ACCURACY_METRICS if m in bdf.columns and pd.notna(br.get(m))}
    return idx


def _collect_recbole_from_evaluation(
    tsv: Path, exp_dir_name: str, exp_meta: dict, metadata: dict, audit: list
) -> list[dict]:
    """Parse *_evaluation.tsv (wide format, one row per model, cols like recall@20).
    Returns one record per (model, cutoff) with all canonical metrics."""
    try:
        df = pd.read_csv(tsv, sep="\t")
    except Exception as e:
        audit.append({"level": "ERROR", "source": str(tsv), "message": f"Read error: {e}"})
        return []
    if df.empty or "model" not in df.columns:
        audit.append({"level": "WARN", "source": str(tsv), "message": "Empty or no 'model' column"})
        return []

    # Discover {cutoff: {canonical_metric: raw_col}} from column names like "recall@20"
    cutoff_metric_map: dict[int, dict[str, str]] = {}
    for col in df.columns:
        m = re.match(r'^(.+)@(\d+)$', col)
        if not m:
            continue
        canonical = _RECBOLE_EVAL_COL.get(m.group(1).lower())
        if canonical is None:
            continue
        cutoff = int(m.group(2))
        cutoff_metric_map.setdefault(cutoff, {})[canonical] = col

    rows = []
    for _, row in df.iterrows():
        model_raw  = str(row.get("model", ""))
        model_norm = normalize_model(model_raw) or model_raw
        for cutoff, metric_col_map in sorted(cutoff_metric_map.items()):
            record = {
                "framework":   "recbole",
                "experiment":  exp_dir_name,
                "source_file": str(tsv),
                "model":       model_norm,
                "full_model":  model_raw,
                "cutoff":      cutoff,
                **exp_meta,
                **{k: metadata.get(k) for k in ["k_user", "k_item"]},
                **{k: metadata.get(k, float("nan")) for k in SPARSITY_OUT if k not in ["k_user", "k_item"]},
            }
            for canonical, raw_col in metric_col_map.items():
                v = row.get(raw_col, float("nan"))
                record[canonical] = float(v) if pd.notna(v) else float("nan")
            rows.append(record)
    return rows


def collect_recbole(recbole_root: Path, processed_root: Path, audit: list) -> list[dict]:
    """Collect rows from results/recbole/{exp}/performance/.
    Priority: *_evaluation.tsv (all metrics in one file).
    Fallback: *_recbole.tsv + *_beyond.tsv (legacy format)."""
    if not recbole_root.exists():
        print(f"[WARN] RecBole results root not found: {recbole_root}")
        return []

    rows = []
    for exp_dir in sorted(recbole_root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("_"):
            continue
        perf_dir = exp_dir / "performance"
        if not perf_dir.exists():
            continue

        try:
            exp_meta = parse_experiment_name(exp_dir.name)
        except ValueError as e:
            audit.append({"level": "ERROR", "source": str(exp_dir), "message": str(e)})
            continue

        metadata = load_metadata(exp_dir.name, processed_root)

        # Index files by model prefix
        eval_files = {
            m.group(1): f
            for f in sorted(perf_dir.glob("*_evaluation.tsv"))
            if (m := re.match(r"^(.+)_evaluation\.tsv$", f.name))
        }
        recbole_files = {
            m.group(1): f
            for f in sorted(perf_dir.glob("*_recbole.tsv"))
            if (m := re.match(r"^(.+)_recbole\.tsv$", f.name))
        }

        # Priority: use *_evaluation.tsv when available
        for fname_model, tsv in eval_files.items():
            rows.extend(_collect_recbole_from_evaluation(
                tsv, exp_dir.name, exp_meta, metadata, audit))
            if fname_model in recbole_files:
                audit.append({
                    "level": "WARN", "source": str(tsv),
                    "message": (f"Both _evaluation.tsv and _recbole.tsv exist for "
                                f"'{fname_model}' — using evaluation file only"),
                })

        # Fallback: models with only legacy *_recbole.tsv
        beyond_idx = _load_beyond_index(perf_dir) if recbole_files else {}
        for fname_model, tsv in recbole_files.items():
            if fname_model in eval_files:
                continue
            try:
                df = pd.read_csv(tsv, sep="\t")
            except Exception as e:
                audit.append({"level": "ERROR", "source": str(tsv), "message": f"Read error: {e}"})
                continue
            if df.empty or "model" not in df.columns or "cutoff" not in df.columns:
                audit.append({"level": "WARN", "source": str(tsv),
                               "message": "Empty or missing 'model'/'cutoff' columns"})
                continue
            for _, row in df.iterrows():
                cutoff = _safe_cutoff(row["cutoff"], str(tsv), audit)
                if cutoff is None:
                    continue
                content_model = str(row.get("model", "")).strip()
                content_norm  = normalize_model(content_model) or content_model
                fname_norm    = normalize_model(fname_model) or fname_model
                if content_norm != fname_norm:
                    audit.append({
                        "level": "WARN", "source": str(tsv),
                        "message": (f"Model mismatch: filename '{fname_norm}', "
                                    f"content '{content_norm}' (using content)"),
                    })
                record = {
                    "framework":   "recbole",
                    "experiment":  exp_dir.name,
                    "source_file": str(tsv),
                    "model":       content_norm,
                    "full_model":  content_model,
                    "cutoff":      cutoff,
                    **exp_meta,
                    **{k: metadata.get(k) for k in ["k_user", "k_item"]},
                    **{k: metadata.get(k, float("nan")) for k in SPARSITY_OUT if k not in ["k_user", "k_item"]},
                }
                for metric in MAIN_METRICS:
                    v = row.get(metric, float("nan"))
                    record[metric] = float(v) if pd.notna(v) else float("nan")
                bdata = beyond_idx.get((content_norm, cutoff), {})
                for metric in BEYOND_ACCURACY_METRICS:
                    record[metric] = bdata.get(metric, float("nan"))
                rows.append(record)

    return rows


# ── series_label ──────────────────────────────────────────────────────────────

def assign_series_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    series_label = model  if model appears in only one framework.
    series_label = '{model} [{Framework}]'  if model in multiple frameworks.
    """
    df = df.copy()
    multi = (
        df.groupby("model")["framework"]
        .nunique()
        .pipe(lambda s: set(s[s > 1].index))
    )
    df["series_label"] = df.apply(
        lambda r: f"{r['model']} [{r['framework'].title()}]"
        if r["model"] in multi
        else r["model"],
        axis=1,
    )
    return df


# ── Duplicate check ───────────────────────────────────────────────────────────

DEDUP_KEY = ["framework", "experiment", "model", "cutoff"]

def check_and_dedup(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    """
    Unique key: framework + experiment + model + cutoff.
    Identical duplicates → keep one, log to audit.
    Conflicting duplicates → raise error.
    """
    dups = df[df.duplicated(subset=DEDUP_KEY, keep=False)]
    if dups.empty:
        return df

    metric_cols = RESULT_METRICS
    to_drop_idx = set()
    errors      = []

    for key, grp in dups.groupby(DEDUP_KEY):
        grp_metrics = grp[metric_cols].reset_index(drop=True)
        first = grp_metrics.iloc[0]
        all_same = all(
            grp_metrics.iloc[i].round(8).equals(first.round(8))
            for i in range(1, len(grp_metrics))
        )
        if all_same:
            audit.append({
                "level": "WARN",
                "source": "; ".join(grp["source_file"].tolist()),
                "message": (f"Exact duplicate: {dict(zip(DEDUP_KEY, key))} "
                            f"({len(grp)} copies) — keeping one"),
            })
            to_drop_idx.update(grp.index[1:].tolist())
        else:
            errors.append(
                f"CONFLICTING duplicate key {dict(zip(DEDUP_KEY, key))}:\n"
                + grp[["source_file"] + metric_cols].to_string()
            )

    if errors:
        raise RuntimeError(
            "Conflicting duplicates found — cannot proceed:\n" + "\n---\n".join(errors)
        )

    return df.drop(index=list(to_drop_idx)).reset_index(drop=True)


# ── Validation ────────────────────────────────────────────────────────────────

def validate_rows(df: pd.DataFrame, audit: list) -> pd.DataFrame:
    """Validate keep_frac, strategy."""
    bad_kf = df["keep_frac"].notna() & ~df["keep_frac"].between(0, 1, inclusive="right")
    for _, row in df[bad_kf].iterrows():
        audit.append({
            "level": "ERROR", "source": row["source_file"],
            "message": f"keep_frac={row['keep_frac']} not in (0,1]",
        })

    bad_strat = df["strategy"].notna() & ~df["strategy"].isin(VALID_STRATEGIES)
    for _, row in df[bad_strat].iterrows():
        audit.append({
            "level": "ERROR", "source": row["source_file"],
            "message": f"strategy='{row['strategy']}' not in {VALID_STRATEGIES}",
        })

    return df


# ── Metadata fallback from reports CSV ───────────────────────────────────────

def try_sparsity_from_reports(df: pd.DataFrame, reports_root: Path, audit: list) -> pd.DataFrame:
    """Fill missing sparsity cols from data/reports/*_summary.csv (fallback).
    Checks ALL sparsity columns for NaN, not just uss.
    """
    if not reports_root.exists():
        return df

    # Fix #2: check all sparsity columns, not just uss
    fallback_cols = [col for col in SPARSITY_OUT if col in df.columns]
    if not fallback_cols:
        return df
    needs_fallback = df[fallback_cols].isna().any(axis=1)
    if not needs_fallback.any():
        return df

    dfs_rep = []
    for csv_path in sorted(reports_root.glob("*_summary.csv")):
        m = re.match(r"^(\w+)_(random|head|tail|base)_summary\.csv$", csv_path.name)
        if not m:
            continue
        try:
            rep = pd.read_csv(csv_path)
            rep["dataset"]  = m.group(1)
            rep["strategy"] = m.group(2)
            rep["keep_frac"] = rep["keep_frac"].astype(float)
            dfs_rep.append(rep)
        except Exception as e:
            audit.append({"level": "WARN", "source": str(csv_path), "message": f"Report read error: {e}"})

    if not dfs_rep:
        return df

    report_df = pd.concat(dfs_rep, ignore_index=True)

    # For base experiments: synthesize from keep_frac≈1.0 (Fix #3: use isclose)
    base_rows = []
    for ds, grp in report_df.groupby("dataset"):
        mask_10 = np.isclose(grp["keep_frac"].astype(float), 1.0, rtol=0, atol=1e-9)
        r = grp[mask_10].head(1).copy()
        if not r.empty:
            r["strategy"] = "base"
            base_rows.append(r)
    if base_rows:
        report_df = pd.concat([report_df] + base_rows, ignore_index=True)

    sparsity_in_report = [c for c in SPARSITY_OUT if c in report_df.columns]
    if not sparsity_in_report:
        return df

    for col in sparsity_in_report:
        if col not in df.columns:
            df[col] = float("nan")

    # Fix #2+#3: create _kf BEFORE drop_duplicates, merge on rounded key
    rep_sub = report_df[["dataset", "strategy", "keep_frac"] + sparsity_in_report].copy()
    rep_sub["_kf"] = rep_sub["keep_frac"].round(8)
    rep_sub = rep_sub.drop_duplicates(subset=["dataset", "strategy", "_kf"])
    df["_kf"] = df["keep_frac"].round(8)

    df_merged = df.merge(
        rep_sub[["dataset", "strategy", "_kf"] + sparsity_in_report],
        on=["dataset", "strategy", "_kf"],
        how="left",
        suffixes=("", "_rep"),
    )
    df_merged.drop(columns=["_kf"], inplace=True)

    for col in sparsity_in_report:
        rep_col = col + "_rep"
        if rep_col in df_merged.columns:
            mask = df_merged[col].isna() & df_merged[rep_col].notna()
            if mask.any():
                df_merged.loc[mask, col] = df_merged.loc[mask, rep_col]
                audit.append({
                    "level": "WARN",
                    "source": "reports fallback",
                    "message": (f"Filled {mask.sum()} missing '{col}' values "
                                "from reports CSV (metadata.json lacked train_metrics)"),
                })
            df_merged.drop(columns=[rep_col], inplace=True)

    return df_merged


# ── Build DataFrames ──────────────────────────────────────────────────────────

_BASE_COLS = [
    "experiment", "dataset", "strategy", "keep_frac",
    "k_user", "k_item", "framework", "model", "series_label", "full_model",
    "cutoff", "source_file",
]


def build_long(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in _BASE_COLS + SPARSITY_OUT + RESULT_METRICS:
        if col not in df.columns:
            df[col] = float("nan") if col in RESULT_METRICS + SPARSITY_OUT else None
    col_order = _BASE_COLS + [c for c in SPARSITY_OUT if c in df.columns] + RESULT_METRICS
    df = df[[c for c in col_order if c in df.columns]]
    df["_so"] = df["strategy"].map(_STRATEGY_ORDER).fillna(99)
    df = (
        df.sort_values(["dataset", "_so", "keep_frac", "framework", "model", "cutoff"])
          .drop(columns=["_so"])
          .reset_index(drop=True)
    )
    return df


def build_wide(long_df: pd.DataFrame, metric_cols: list[str], sheet_name: str) -> pd.DataFrame:
    """Pivot metric_cols over cutoff. Joins k_user/k_item/sparsity after pivot."""
    pivot_index    = ["experiment", "dataset", "strategy", "keep_frac",
                      "framework", "model", "series_label"]
    meta_join_cols = ["k_user", "k_item"] + [c for c in SPARSITY_OUT if c in long_df.columns]

    meta_per_exp = (
        long_df[["experiment"] + meta_join_cols]
        .drop_duplicates(subset=["experiment"])
        .set_index("experiment")
    )

    wide = long_df.pivot_table(
        index=pivot_index,
        columns="cutoff",
        values=metric_cols,
        aggfunc="first",
    )
    wide.columns = [f"{m}@{c}" for m, c in wide.columns]
    wide = wide.reset_index()
    wide = wide.join(meta_per_exp, on="experiment", how="left")

    cutoffs     = sorted(long_df["cutoff"].dropna().unique().astype(int))
    metric_flat = [f"{m}@{c}" for c in cutoffs for m in metric_cols]
    final_cols  = pivot_index + ["k_user", "k_item"] + \
                  [c for c in SPARSITY_OUT if c in wide.columns] + \
                  [c for c in metric_flat if c in wide.columns]
    wide = wide[[c for c in final_cols if c in wide.columns]]

    wide["_so"] = wide["strategy"].map(_STRATEGY_ORDER).fillna(99)
    wide = (
        wide.sort_values(["dataset", "_so", "keep_frac", "framework", "model"])
            .drop(columns=["_so"])
            .reset_index(drop=True)
    )
    return wide


def _find_processed_experiments(processed_root: Path) -> list[tuple[str, dict]]:
    """
    Scan data/processed/ for parseable experiment folders.
    Returns list of (folder_name, parsed_meta) tuples.
    Folders starting with '_' or unparseable names are silently skipped.
    """
    if not processed_root.exists():
        return []
    result = []
    for d in sorted(processed_root.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        try:
            meta = parse_experiment_name(d.name)
            result.append((d.name, meta))
        except ValueError:
            pass
    return result


def build_coverage(
    long_df: pd.DataFrame,
    exp_models: list[str],
    exp_cutoffs: list[int],
    cli_overrides: dict[str, list[str]] | None = None,
    all_experiments: list[tuple[str, dict]] | None = None,
) -> pd.DataFrame:
    """
    Build experiment × (framework, model, cutoff) coverage table.

    Framework-per-model priority (Fix #1):
      1. CLI override (--model-frameworks)
      2. Frameworks actually detected in long_df data
      3. DEFAULT_MODEL_FRAMEWORKS fallback (only if model not detected at all)

    Experiments (Fix #4): union of long_df experiments + all_experiments from
    data/processed/. Experiments with zero results still appear (all cells "-").
    """
    if cli_overrides is None:
        cli_overrides = {}

    # Build detected frameworks per model from actual data
    detected_by_model: dict[str, list[str]] = {}
    for (fw, model), _ in long_df.groupby(["framework", "model"]):
        if model in exp_models:
            detected_by_model.setdefault(model, [])
            if fw not in detected_by_model[model]:
                detected_by_model[model].append(fw)

    # Resolve expected (framework, model) pairs with priority order
    expected_pairs: set[tuple[str, str]] = set()
    for model in exp_models:
        if model in cli_overrides:
            fws = cli_overrides[model]
        elif model in detected_by_model:
            fws = detected_by_model[model]
        else:
            fws = DEFAULT_MODEL_FRAMEWORKS.get(model, [])
        for fw in fws:
            expected_pairs.add((fw, model))

    present = set(
        zip(long_df["experiment"], long_df["framework"], long_df["model"], long_df["cutoff"])
    )

    # Build full experiment set: from results + from processed/ (Fix #4)
    exp_meta_lookup: dict[str, dict] = {}
    for exp in long_df["experiment"].unique():
        try:
            exp_meta_lookup[exp] = parse_experiment_name(exp)
        except ValueError:
            exp_meta_lookup[exp] = {"dataset": exp, "strategy": None, "keep_frac": None}
    if all_experiments:
        for exp_name, meta in all_experiments:
            if exp_name not in exp_meta_lookup:
                exp_meta_lookup[exp_name] = meta

    experiments  = sorted(exp_meta_lookup)
    sorted_pairs = sorted(expected_pairs)

    records = []
    for exp in experiments:
        exp_meta = exp_meta_lookup[exp]
        for (fw, model) in sorted_pairs:
            row = {
                "experiment": exp, "framework": fw, "model": model,
                "dataset":    exp_meta["dataset"],
                "strategy":   exp_meta["strategy"],
                "keep_frac":  exp_meta["keep_frac"],
            }
            has_any = False
            for cutoff in exp_cutoffs:
                val = "OK" if (exp, fw, model, cutoff) in present else "-"
                row[f"cutoff_{cutoff}"] = val
                if val == "OK":
                    has_any = True
            row["any_present"] = "OK" if has_any else "-"
            records.append(row)

    cov = pd.DataFrame(records)
    cov["_so"] = cov["strategy"].map(_STRATEGY_ORDER).fillna(99)
    cov = (
        cov.sort_values(["dataset", "_so", "keep_frac", "framework", "model"])
           .drop(columns=["_so"])
           .reset_index(drop=True)
    )
    return cov


# ── Excel writer ──────────────────────────────────────────────────────────────

def _autofit(ws, max_width=40):
    for col_cells in ws.columns:
        max_len = max(
            (len(str(c.value)) if c.value is not None else 0) for c in col_cells
        )
        ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, max_width)
    ws.freeze_panes = "A2"


def write_excel(sheets: dict[str, pd.DataFrame], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            _autofit(writer.sheets[name])
    print(f"\nWritten: {output_path}")
    for name, df in sheets.items():
        print(f"  {name:<28s}: {df.shape[0]:>5} rows × {df.shape[1]} cols")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_model_frameworks(specs: list[str]) -> dict[str, list[str]]:
    """Parse CLI 'ModelName:fw1,fw2' specs into dict (no default merging)."""
    result: dict[str, list[str]] = {}
    for spec in specs or []:
        if ":" not in spec:
            raise argparse.ArgumentTypeError(
                f"--model-frameworks: invalid spec '{spec}' — expected 'ModelName:fw1,fw2'"
            )
        model, fws = spec.split(":", 1)
        result[model.strip()] = [f.strip() for f in fws.split(",")]
    return result


def parse_args():
    p = argparse.ArgumentParser(description="Build unified performance summary Excel")
    p.add_argument("--elliot-results-root",  default=str(DEFAULT_ELLIOT_ROOT))
    p.add_argument("--recbole-results-root", default=str(DEFAULT_RECBOLE_ROOT))
    p.add_argument("--processed-root",       default=str(DEFAULT_PROCESSED))
    p.add_argument("--reports-root",         default=str(DEFAULT_REPORTS))
    p.add_argument("--output",               default=str(DEFAULT_OUTPUT))
    p.add_argument("--expected-models",      nargs="+", default=EXPECTED_MODELS)
    p.add_argument("--expected-cutoffs",     nargs="+", type=int, default=EXPECTED_CUTOFFS)
    p.add_argument(
        "--model-frameworks", nargs="*", default=[],
        metavar="MODEL:FW1,FW2",
        help=(
            "Override model→framework mapping for coverage. "
            "E.g. 'ItemKNN:elliot,recbole' BPR:recbole'. "
            "Default: VSM/FunkSVD→elliot, ItemKNN/BPR→recbole."
        ),
    )
    p.add_argument(
        "--allow-audit-errors",
        action="store_true",
        default=False,
        help="Write workbook even when audit ERRORs exist (default: exit on first error batch)",
    )
    return p.parse_args()


def main():
    args           = parse_args()
    elliot_root    = Path(args.elliot_results_root)
    recbole_root   = Path(args.recbole_results_root)
    processed_root = Path(args.processed_root)
    reports_root   = Path(args.reports_root)
    output_path    = Path(args.output)
    exp_models     = args.expected_models
    exp_cutoffs    = args.expected_cutoffs

    # Parse CLI model-framework overrides (only user-specified entries)
    try:
        cli_overrides = _parse_model_frameworks(args.model_frameworks or [])
    except argparse.ArgumentTypeError as e:
        sys.exit(str(e))

    audit: list[dict] = []

    # ── Collect ────────────────────────────────────────────────────────────────
    print("Collecting Elliot performance files …")
    elliot_rows = collect_elliot(elliot_root, processed_root, audit)
    print(f"  {len(elliot_rows)} rows from Elliot")

    print("Collecting RecBole performance files …")
    recbole_rows = collect_recbole(recbole_root, processed_root, audit)
    print(f"  {len(recbole_rows)} rows from RecBole")

    if not elliot_rows and not recbole_rows:
        sys.exit("ERROR: No performance data found from either framework.")

    # ── Merge and validate ─────────────────────────────────────────────────────
    long_df = build_long(elliot_rows + recbole_rows)
    long_df = validate_rows(long_df, audit)
    long_df = check_and_dedup(long_df, audit)
    long_df = assign_series_labels(long_df)
    long_df = try_sparsity_from_reports(long_df, reports_root, audit)

    # ── Audit error check (Fix #6) ─────────────────────────────────────────────
    errors = [a for a in audit if a.get("level") == "ERROR"]
    warns  = [a for a in audit if a.get("level") == "WARN"]
    print(f"\nAudit: {len(errors)} errors, {len(warns)} warnings")

    if errors:
        print("\nAudit ERRORs (must be resolved before writing workbook):")
        for e in errors:
            print(f"  ERROR: {e['message']}")
            print(f"         source: {e.get('source','')}")
        if not args.allow_audit_errors:
            sys.exit(
                f"\nAborted: {len(errors)} audit error(s) found. "
                "Fix data issues or pass --allow-audit-errors to override."
            )
        print("[WARN] --allow-audit-errors set: proceeding despite errors.")

    for w in warns[:10]:
        print(f"  WARN : {w['message']}")
    if len(warns) > 10:
        print(f"  … and {len(warns)-10} more warnings (see data_quality_audit sheet)")

    print(f"\nTotal rows: {len(long_df)}")
    for (fw, model), grp in long_df.groupby(["framework", "model"]):
        print(f"  [{fw:8s}] {model:10s}: {len(grp):3d} rows, "
              f"{grp['experiment'].nunique()} experiments, "
              f"cutoffs={sorted(grp['cutoff'].unique().tolist())}")

    # ── Scan processed/ for expected experiments (Fix #4) ─────────────────────
    all_experiments = _find_processed_experiments(processed_root)
    processed_names = {name for name, _ in all_experiments}
    result_names    = set(long_df["experiment"].unique())
    no_result_exps  = sorted(processed_names - result_names)
    if no_result_exps:
        print(f"\n[WARN] {len(no_result_exps)} processed experiments have NO performance results:")
        for exp in no_result_exps:
            print(f"  NO RESULTS: {exp}")

    # ── Build sheets ───────────────────────────────────────────────────────────
    overall_wide      = build_wide(long_df, MAIN_METRICS,            "overall_wide")
    cold_u1_wide      = build_wide(long_df, COLD_U1_METRICS,         "cold_u1_wide")
    beyond_wide       = build_wide(long_df, BEYOND_ACCURACY_METRICS, "beyond_accuracy_wide")
    pop_groups_wide   = build_wide(long_df, POP_METRICS,             "popularity_groups_wide")
    user_groups_wide  = build_wide(long_df, USER_METRICS,            "user_groups_wide")
    coverage          = build_coverage(long_df, exp_models, exp_cutoffs,
                                       cli_overrides=cli_overrides,
                                       all_experiments=all_experiments)
    audit_df          = pd.DataFrame(audit) if audit else pd.DataFrame(
        columns=["level", "source", "message"])

    # Coverage gaps (only expected pairs, experiments from processed/ included)
    missing_any = coverage[coverage["any_present"] == "-"]
    if not missing_any.empty:
        print(f"\nCoverage gaps ({len(missing_any)} expected experiment×framework×model combinations):")
        for _, r in missing_any.iterrows():
            print(f"  MISSING: [{r['framework']}] {r['experiment']} / {r['model']}")

    # ── Write Excel ────────────────────────────────────────────────────────────
    sheets = {
        "long_all_results":      long_df,
        "overall_wide":          overall_wide,
        "cold_u1_wide":          cold_u1_wide,
        "beyond_accuracy_wide":  beyond_wide,
        "popularity_groups_wide": pop_groups_wide,
        "user_groups_wide":      user_groups_wide,
        "experiment_coverage":   coverage,
        "data_quality_audit":    audit_df,
    }
    write_excel(sheets, output_path)


if __name__ == "__main__":
    main()
