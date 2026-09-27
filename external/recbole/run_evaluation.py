import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
np.float = float

_PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR    = _PROJECT_ROOT / "scripts"
_ARTIFACTS_ROOT = _PROJECT_ROOT / "results" / "selected_artifacts"

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from selected_artifact import read_artifact, print_artifact

# ---------------------------------------------------------------------------
# Metrics added on top of the checkpoint's original metric list
# ---------------------------------------------------------------------------

_pop_metrics = [
    'PopRecall_pop1',     'PopRecall_pop2_5',   'PopRecall_pop6_10',
    'PopRecall_pop11_20', 'PopRecall_pop21_40', 'PopRecall_pop41plus',
    'PopNDCG_pop1',       'PopNDCG_pop2_5',     'PopNDCG_pop6_10',
    'PopNDCG_pop11_20',   'PopNDCG_pop21_40',   'PopNDCG_pop41plus',
]
_user_metrics = [
    'UserRecall_u1',     'UserRecall_u2_5',   'UserRecall_u6_10',
    'UserRecall_u11_20', 'UserRecall_u21_40', 'UserRecall_u41plus',
    'UserNDCG_u1',       'UserNDCG_u2_5',     'UserNDCG_u6_10',
    'UserNDCG_u11_20',   'UserNDCG_u21_40',   'UserNDCG_u41plus',
]
_beyond_metrics = ['ItemCoverage', 'AveragePopularity', 'GiniIndex', 'TailPercentage']


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_beyond_accuracy_result(dataset_name, model_name, test_result):
    """Save beyond-accuracy metrics per cutoff to *_beyond.tsv using canonical column names.

    RecBole native → canonical:
      itemcoverage      → ItemCoverage
      averagepopularity → AveragePopularity
      giniindex         → Gini
      tailpercentage    → TailPercentage
    """
    _beyond_map = {
        'itemcoverage':      'ItemCoverage',
        'averagepopularity': 'AveragePopularity',
        'giniindex':         'Gini',
        'tailpercentage':    'TailPercentage',
    }
    _beyond_cols = ['ItemCoverage', 'AveragePopularity', 'Gini', 'TailPercentage']

    by_cutoff: dict = {}
    for key, val in test_result.items():
        if '@' not in str(key):
            continue
        metric_raw, cutoff_str = str(key).rsplit('@', 1)
        col = _beyond_map.get(metric_raw.strip().lower())
        if col is None:
            continue
        try:
            cutoff = int(cutoff_str.strip())
        except ValueError:
            continue
        by_cutoff.setdefault(cutoff, {})[col] = float(val)

    if not by_cutoff:
        return

    out_dir  = _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_beyond.tsv"

    rows = []
    for cutoff in sorted(by_cutoff):
        row = {'model': model_name, 'cutoff': cutoff}
        for col in _beyond_cols:
            row[col] = by_cutoff[cutoff].get(col, float('nan'))
        rows.append(row)

    with open(out_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'cutoff'] + _beyond_cols, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)

    rel = out_file.relative_to(_PROJECT_ROOT)
    print(f"\n[evaluation] Beyond-accuracy metrics saved:\n  {rel}")


def _save_evaluation_result(dataset_name, model_name, artifact_path, test_result, path_field="checkpoint"):
    """Save the wide-format *_evaluation.tsv row.

    path_field names the column holding artifact_path: "checkpoint" for native
    RecBole checkpoint evaluation (unchanged, for backward compatibility --
    nothing downstream reads this column by name, but the value must stay
    honest), "artifact" for external precomputed-recommendation evaluation
    (never call a recommendation TSV a "checkpoint").
    """
    out_dir  = _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_evaluation.tsv"

    row = {"dataset": dataset_name, "model": model_name, path_field: artifact_path,
           **test_result}

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    rel = out_file.relative_to(_PROJECT_ROOT)
    print(f"\n[evaluation] Saved:\n  {rel}")


# ---------------------------------------------------------------------------
# Core evaluation for one checkpoint (framework="recbole", artifact_type="checkpoint")
# ---------------------------------------------------------------------------

def _run_checkpoint(
    checkpoint_path: str,
    overwrite: bool = False,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> str:
    # Fast skip when caller already knows dataset/model (avoids loading the checkpoint)
    if dataset_name and model_name:
        out_file = (
            _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
            / f"{model_name}_evaluation.tsv"
        )
        if out_file.exists() and not overwrite:
            print(f"  [SKIP] {out_file.relative_to(_PROJECT_ROOT)}  (use --overwrite)")
            return "skip"

    config, model, _, train_data, _, test_data = \
        load_data_and_model(model_file=checkpoint_path)

    dataset_name = config["dataset"]
    model_name   = config["model"]

    # Fallback skip check for the -f path where dataset/model were unknown before load
    out_file = (
        _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
        / f"{model_name}_evaluation.tsv"
    )
    if out_file.exists() and not overwrite:
        print(f"  [SKIP] {out_file.relative_to(_PROJECT_ROOT)}  (use --overwrite)")
        return "skip"

    # tail_ratio=0.1 → bottom 10% least-popular items (fraction branch in TailPercentage)
    config['tail_ratio'] = 0.1
    _additional = _pop_metrics + _user_metrics + _beyond_metrics
    config['metrics'] = list(dict.fromkeys(list(config['metrics']) + _additional))

    trainer = Trainer(config, model)
    trainer.eval_collector.data_collect(train_data)

    result = trainer.evaluate(
        test_data,
        load_best_model=False,
        show_progress=config["show_progress"],
    )
    print(result)

    _save_evaluation_result(dataset_name, model_name, checkpoint_path, result)
    _save_beyond_accuracy_result(dataset_name, model_name, result)
    return "ok"


# ---------------------------------------------------------------------------
# External recommendation evaluation (framework="librecommender", artifact_type="recommendations")
#
# Core principle: do NOT reimplement RecBole's metric formulas. This path only
# constructs the same DataStruct resources (rec.items, rec.topk, rec.topk_<pop
# group>, data.user_ids, plus the TRAIN-derived data.num_items/count_items/
# count_users/pop_groups) that native checkpoint evaluation feeds into the
# same Collector/Evaluator/metric classes -- see collector.py::eval_batch_collect
# and metrics.py for the exact construction this mirrors.
# ---------------------------------------------------------------------------

_NEUTRAL_CONFIG_FILE = _PROJECT_ROOT / "configs" / "recbole" / "bpr.yml"
_RECBOLE_DATA_ROOT   = _PROJECT_ROOT / "data" / "recbole"

_POP_GROUP_KEYS = ["pop1", "pop2_5", "pop6_10", "pop11_20", "pop21_40", "pop41plus"]


def _build_external_eval_context(dataset_name: str):
    """Build (config, dataset, train_data, test_data) with the SAME official
    RecBole APIs used for checkpoint loading, but without any model checkpoint.

    configs/recbole/bpr.yml is used purely as a neutral data/eval schema
    (benchmark train/valid/test filenames, USER_ID_FIELD/ITEM_ID_FIELD, topk,
    base metrics list, metric_decimal_place). BPR itself is never instantiated
    or trained -- only Config/create_dataset/data_preparation are called, so
    this reconstructs the identical token2id mappings, TRAIN item/user
    counters, and TEST ground truth as native checkpoint evaluation on this
    dataset (it reads the exact same data/recbole/<dataset>/*.inter files with
    the exact same schema).
    """
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed, init_logger

    if not _NEUTRAL_CONFIG_FILE.exists():
        raise FileNotFoundError(f"Neutral schema config not found: {_NEUTRAL_CONFIG_FILE}")

    config = Config(
        model="BPR",
        dataset=dataset_name,
        config_file_list=[str(_NEUTRAL_CONFIG_FILE)],
        config_dict={
            "data_path": str(_RECBOLE_DATA_ROOT) + "/",
            "use_gpu": False,           # no model computation happens in this path
            "save_dataset": False,
            "save_dataloaders": False,
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, _valid_data, test_data = data_preparation(config, dataset)
    return config, dataset, train_data, test_data


def _load_external_recommendations(path: Path, topk_needed: int) -> dict[str, list[str]]:
    """Read + validate a precomputed recommendation TSV.

    Format: user_id<TAB>item_id<TAB>score, no header. Raw IDs are read as
    strings. Fails loudly (RuntimeError) on any malformed input -- never
    silently drops or fixes rows.

    Returns {raw_user_id: [raw_item_id, ...]} where each list is already
    score-descending (as validated) and has length <= topk_needed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Recommendation TSV not found: {path}")

    df = pd.read_csv(
        path, sep="\t", header=None, dtype=str,
        names=["user_id", "item_id", "score"],
        keep_default_na=False, na_values=[""],
    )
    if df.shape[1] != 3:
        raise RuntimeError(f"{path}: expected exactly 3 columns, got {df.shape[1]}")

    n_null_user = int(df["user_id"].isna().sum())
    n_null_item = int(df["item_id"].isna().sum())
    if n_null_user or n_null_item:
        raise RuntimeError(
            f"{path}: {n_null_user} null user_id, {n_null_item} null item_id -- refusing to proceed."
        )

    scores = pd.to_numeric(df["score"], errors="coerce")
    n_bad_score = int(scores.isna().sum())
    if n_bad_score:
        bad_rows = df.loc[scores.isna()].head(5)
        raise RuntimeError(
            f"{path}: {n_bad_score} non-numeric score value(s). Sample:\n{bad_rows}"
        )
    scores = scores.astype(float)
    if not np.isfinite(scores.to_numpy()).all():
        n_nonfinite = int((~np.isfinite(scores.to_numpy())).sum())
        raise RuntimeError(f"{path}: {n_nonfinite} non-finite score value(s).")
    df["score"] = scores

    dup_mask = df.duplicated(subset=["user_id", "item_id"])
    if dup_mask.any():
        sample = df.loc[dup_mask, ["user_id", "item_id"]].head(10).values.tolist()
        raise RuntimeError(
            f"{path}: {int(dup_mask.sum())} duplicate (user,item) row(s). Sample: {sample}"
        )

    result: dict[str, list[str]] = {}
    bad_order_users = []
    too_long_users = []
    for uid, g in df.groupby("user_id", sort=False):
        items = g["item_id"].tolist()
        s = g["score"].tolist()
        if s != sorted(s, reverse=True):
            bad_order_users.append(uid)
            continue
        if len(items) > topk_needed:
            too_long_users.append((uid, len(items)))
            continue
        result[uid] = items

    if bad_order_users:
        raise RuntimeError(
            f"{path}: {len(bad_order_users)} user(s) have non-descending scores. "
            f"Sample: {bad_order_users[:10]}"
        )
    if too_long_users:
        raise RuntimeError(
            f"{path}: {len(too_long_users)} user(s) exceed topk_needed={topk_needed} rows. "
            f"Sample (user, n_rows): {too_long_users[:10]}"
        )

    print(f"  [OK] loaded external recommendations: {path}\n       users={len(result):,}")
    return result


def _build_external_data_struct(config, dataset, train_data, test_data, ext_recs, topk_needed):
    """Construct the DataStruct that native checkpoint evaluation would produce,
    using the supplied external ranking in place of model scores.

    TRAIN-derived resources (data.num_items, data.count_items, data.count_users,
    data.pop_groups) are obtained by running the SAME Collector.data_collect()
    used by native evaluation on the SAME train_data -- never recomputed here.

    rec.items / rec.topk / rec.topk_<pop group> / data.user_ids are built by
    hand from the external ranking, following exactly the same construction
    Collector.eval_batch_collect() uses (see collector.py): rec.items is the
    per-user internal item id sequence in ranked order; rec.topk is a
    (hit-at-rank..., pos_len) row per user where hits are against TEST ground
    truth only; rec.topk_<group> is the same but with GT filtered to that
    TRAIN-popularity group -- the ranking itself is never re-ranked or
    filtered, only which GT positions count as a "hit" changes.

    Test-user row order is RecBole's own canonical order: test_data.uid_list,
    which FullSortEvalDataLoader builds by sorting internal user ids ascending
    (see general_dataloader.py) -- never assumed from raw-ID sort order.
    """
    from recbole.evaluator.collector import Collector
    import torch

    uid_field = config["USER_ID_FIELD"]
    iid_field = config["ITEM_ID_FIELD"]

    collector = Collector(config)
    collector.data_collect(train_data)
    data_struct = collector.data_struct

    if "data.pop_groups" not in data_struct:
        raise RuntimeError(
            "data.pop_groups was not collected -- config['metrics'] must include the "
            "Pop* group metrics before building the external DataStruct."
        )
    pop_groups = data_struct.get("data.pop_groups")

    test_uid_list = test_data.uid_list.numpy()
    uid2positive_item = test_data.uid2positive_item
    n_users = len(test_uid_list)

    rec_items = torch.zeros((n_users, topk_needed), dtype=torch.int64)
    rec_topk = torch.zeros((n_users, topk_needed + 1), dtype=torch.int64)
    data_user_ids = torch.zeros((n_users,), dtype=torch.int64)
    rec_topk_group = {
        g: torch.zeros((n_users, topk_needed + 1), dtype=torch.int64) for g in _POP_GROUP_KEYS
    }

    missing_tsv_users, empty_gt_users, short_row_users = [], [], []
    for row, uid in enumerate(test_uid_list):
        uid = int(uid)
        raw_user = str(dataset.id2token(uid_field, uid))
        data_user_ids[row] = uid

        gt_items = set(int(x) for x in uid2positive_item[uid].tolist())
        if not gt_items:
            empty_gt_users.append(raw_user)
            continue

        if raw_user not in ext_recs:
            missing_tsv_users.append(raw_user)
            continue
        raw_items = ext_recs[raw_user]
        # scripts/run_two_tower.py may WARN (not fail) when a user's candidate
        # pool after train+valid masking is smaller than topk, and emit fewer
        # than topk_needed rows for that user. This evaluator intentionally
        # does NOT tolerate that here: metric parity with RecBole's native
        # full Top-50 evaluation (see the parity test) requires every user's
        # rec.topk/rec.items row to be exactly topk_needed positions wide, and
        # there is no safe way to fill the gap (no fake item, no internal id
        # 0 -- see the padding/unknown-id check below) without silently
        # biasing coverage/Gini/etc. A short row is therefore a signal to go
        # investigate that experiment's recommendation generation, not
        # something to paper over in the shared evaluator.
        if len(raw_items) != topk_needed:
            short_row_users.append((raw_user, len(raw_items)))
            continue

        internal_items = []
        for it in raw_items:
            try:
                iid = int(dataset.token2id(iid_field, it))
            except ValueError:
                raise RuntimeError(
                    f"user={raw_user!r} recommended item={it!r} is unknown to the "
                    f"RecBole item mapping for dataset={config['dataset']!r}."
                )
            if iid == 0:
                raise RuntimeError(
                    f"user={raw_user!r} recommended item={it!r} maps to internal id 0 "
                    f"(padding/unknown) -- refusing to include it as a recommendation."
                )
            internal_items.append(iid)

        rec_items[row, :] = torch.tensor(internal_items, dtype=torch.int64)

        hit = torch.tensor([1 if iid in gt_items else 0 for iid in internal_items], dtype=torch.int64)
        rec_topk[row, :topk_needed] = hit
        rec_topk[row, topk_needed] = len(gt_items)

        for g in _POP_GROUP_KEYS:
            gt_group = gt_items & pop_groups[g]
            hit_g = torch.tensor(
                [1 if iid in gt_group else 0 for iid in internal_items], dtype=torch.int64
            )
            rec_topk_group[g][row, :topk_needed] = hit_g
            rec_topk_group[g][row, topk_needed] = len(gt_group)

    if empty_gt_users:
        raise RuntimeError(
            f"{len(empty_gt_users)} TEST user(s) have empty ground truth (should never "
            f"happen -- every user in test_data.uid_list must have >=1 TEST interaction). "
            f"Sample: {empty_gt_users[:10]}"
        )
    if missing_tsv_users:
        raise RuntimeError(
            f"{len(missing_tsv_users)} TEST user(s) missing from the recommendation TSV. "
            f"Sample: {missing_tsv_users[:10]}"
        )
    if short_row_users:
        raise RuntimeError(
            f"{len(short_row_users)} TEST user(s) have fewer than topk_needed={topk_needed} "
            f"recommendation rows. Exact Top-{topk_needed} is required for shared RecBole "
            f"metric parity (config topk={config['topk']}); this evaluator will not pad with "
            f"a fake item or internal id 0. If the generating script (e.g. run_two_tower.py) "
            f"reported short-candidate users for this experiment, investigate the candidate "
            f"pool there instead of padding here. Sample (user, n_rows): {short_row_users[:10]}"
        )

    extra_tsv_users = set(ext_recs) - {str(dataset.id2token(uid_field, int(u))) for u in test_uid_list}
    if extra_tsv_users:
        raise RuntimeError(
            f"{len(extra_tsv_users)} user(s) in the recommendation TSV are not TEST users. "
            f"Sample: {sorted(extra_tsv_users)[:10]}"
        )

    if rec_items.shape[0] != rec_topk.shape[0] or rec_items.shape[0] != data_user_ids.shape[0]:
        raise RuntimeError(
            f"Shape mismatch: rec.items={tuple(rec_items.shape)} rec.topk={tuple(rec_topk.shape)} "
            f"data.user_ids={tuple(data_user_ids.shape)}"
        )
    for g, mat in rec_topk_group.items():
        if mat.shape != rec_topk.shape:
            raise RuntimeError(f"Shape mismatch: rec.topk_{g}={tuple(mat.shape)} != rec.topk={tuple(rec_topk.shape)}")

    data_struct.set("rec.items", rec_items)
    data_struct.set("rec.topk", rec_topk)
    data_struct.set("data.user_ids", data_user_ids)
    for g in _POP_GROUP_KEYS:
        data_struct.set(f"rec.topk_{g}", rec_topk_group[g])

    print(f"  [OK] external DataStruct built: {n_users:,} test users, topk_needed={topk_needed}")
    return data_struct


def _run_external_recommendations(
    tsv_path: str,
    overwrite: bool = False,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> str:
    if not dataset_name or not model_name:
        raise ValueError("_run_external_recommendations requires dataset_name and model_name.")

    out_file = (
        _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
        / f"{model_name}_evaluation.tsv"
    )
    if out_file.exists() and not overwrite:
        print(f"  [SKIP] {out_file.relative_to(_PROJECT_ROOT)}  (use --overwrite)")
        return "skip"

    tsv_path = Path(tsv_path)
    config, dataset, train_data, test_data = _build_external_eval_context(dataset_name)

    # Same additions, same tail_ratio, same ordering-before-Collector/Evaluator
    # construction as native _run_checkpoint() -- Register reads config['metrics']
    # once at Collector/Evaluator construction time.
    config['tail_ratio'] = 0.1
    _additional = _pop_metrics + _user_metrics + _beyond_metrics
    config['metrics'] = list(dict.fromkeys(list(config['metrics']) + _additional))

    max_k = max(config["topk"])
    ext_recs = _load_external_recommendations(tsv_path, max_k)

    data_struct = _build_external_data_struct(
        config, dataset, train_data, test_data, ext_recs, max_k
    )

    from recbole.evaluator.evaluator import Evaluator
    evaluator = Evaluator(config)
    result = evaluator.evaluate(data_struct)
    print(result)

    _save_evaluation_result(dataset_name, model_name, str(tsv_path), result, path_field="artifact")
    _save_beyond_accuracy_result(dataset_name, model_name, result)
    return "ok"


# ---------------------------------------------------------------------------
# Dataset/model discovery for --all
# ---------------------------------------------------------------------------

# Explicitly enumerated -- do NOT accept arbitrary framework/artifact_type
# combinations. Extend this set (and _dispatch_artifact below) when a new
# evaluator input type is supported.
_SUPPORTED_FRAMEWORK_ARTIFACT = {
    ("recbole", "checkpoint"),
    ("librecommender", "recommendations"),
}


def _discover_pairs(filter_str: str | None, model_filter: str | None) -> list[tuple[str, str]]:
    """Return supported (dataset, model) pairs from results/selected_artifacts/."""
    if not _ARTIFACTS_ROOT.exists():
        raise SystemExit(f"ERROR: {_ARTIFACTS_ROOT} does not exist — run recbole_run.py first.")
    pairs = []
    for ds_dir in sorted(_ARTIFACTS_ROOT.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        if filter_str and filter_str.lower() not in dataset.lower():
            continue
        for artifact_file in sorted(ds_dir.glob("*.json")):
            try:
                meta = json.loads(artifact_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (meta.get("framework"), meta.get("artifact_type")) not in _SUPPORTED_FRAMEWORK_ARTIFACT:
                continue
            model = artifact_file.stem
            if model_filter and model.lower() != model_filter.lower():
                continue
            pairs.append((dataset, model))
    if not pairs:
        raise SystemExit("No artifact pairs found matching the given filters.")
    return pairs


def _dispatch_artifact(meta: dict, overwrite: bool, dataset_name: str, model_name: str) -> str:
    """Route a selected-artifact record to the matching evaluation path."""
    key = (meta.get("framework"), meta.get("artifact_type"))
    if key not in _SUPPORTED_FRAMEWORK_ARTIFACT:
        raise SystemExit(
            f"Unsupported artifact for dataset={dataset_name!r} model={model_name!r}: "
            f"framework={meta.get('framework')!r} artifact_type={meta.get('artifact_type')!r}. "
            f"Supported: {sorted(_SUPPORTED_FRAMEWORK_ARTIFACT)}"
        )
    if key == ("recbole", "checkpoint"):
        return _run_checkpoint(
            meta["_resolved_path"], overwrite=overwrite,
            dataset_name=dataset_name, model_name=model_name,
        )
    else:  # ("librecommender", "recommendations")
        return _run_external_recommendations(
            meta["_resolved_path"], overwrite=overwrite,
            dataset_name=dataset_name, model_name=model_name,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run RecBole post-hoc evaluation (group + beyond-accuracy metrics)."
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-f", "--filetrained",
        metavar="CHECKPOINT",
        help="Path to a trained model checkpoint (.pth). Ignores --dataset/--model.",
    )
    source.add_argument(
        "--dataset",
        metavar="DATASET",
        help="Dataset name for metadata auto-lookup (requires --model).",
    )
    source.add_argument(
        "--all",
        action="store_true",
        help="Run on all (dataset, model) pairs found in results/selected_artifacts/.",
    )

    # NOT part of the `source` mutually exclusive group: -e is a modifier on
    # the --dataset/--model mode (it swaps where the recommendation TSV comes
    # from -- selected-artifact lookup vs. a directly-supplied path), not an
    # independent source mode of its own. Putting it in `source` would make
    # argparse reject `--dataset X --model Y -e file.tsv` outright, which is
    # exactly the documented debug workflow -- see the validation below.
    parser.add_argument(
        "-e", "--external-file",
        metavar="RECOMMENDATION_TSV",
        help=(
            "Debug-only: evaluate a precomputed recommendation TSV directly, bypassing "
            "selected-artifact metadata. Requires --dataset and --model, and is "
            "incompatible with -f/--filetrained and --all. Prefer letting "
            "--dataset/--model or --all resolve artifacts via selected_artifact.py."
        ),
    )

    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Model name. Required with --dataset; optional filter with --all.",
    )
    parser.add_argument(
        "--filter",
        metavar="TEXT",
        help="(with --all) only datasets whose name contains TEXT.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run even if *_evaluation.tsv already exists.",
    )

    args = parser.parse_args()

    if args.filter and not args.all:
        parser.error("--filter can only be used with --all.")
    if args.filetrained and args.model:
        parser.error("--model cannot be used with -f/--filetrained.")
    if args.external_file and args.filetrained:
        parser.error("-e/--external-file cannot be used with -f/--filetrained.")
    if args.external_file and args.all:
        parser.error("-e/--external-file cannot be used with --all.")
    if args.external_file and not (args.dataset and args.model):
        parser.error("-e/--external-file requires --dataset and --model.")

    # ── Single checkpoint via -f ──────────────────────────────────────────────
    if args.filetrained:
        _run_checkpoint(args.filetrained, overwrite=args.overwrite)
        return

    # ── Debug-only: direct recommendation TSV via -e ─────────────────────────
    if args.external_file:
        _run_external_recommendations(
            args.external_file, overwrite=args.overwrite,
            dataset_name=args.dataset, model_name=args.model,
        )
        return

    # ── Single dataset/model ──────────────────────────────────────────────────
    if args.dataset:
        if not args.model:
            parser.error("--dataset requires --model.")
        meta = read_artifact(args.dataset, args.model)
        print_artifact(meta)
        _dispatch_artifact(meta, args.overwrite, args.dataset, args.model)
        return

    # ── --all ─────────────────────────────────────────────────────────────────
    pairs = _discover_pairs(args.filter, args.model)
    print(f"Found {len(pairs)} artifact(s) to evaluate:")
    for ds, m in pairs:
        print(f"  {ds}  /  {m}")
    print()

    success = skip = error = 0
    for dataset, model_name in pairs:
        print(f"\n{'='*60}")
        print(f"dataset={dataset}  model={model_name}")
        print("="*60)
        try:
            meta = read_artifact(dataset, model_name)
            print_artifact(meta)
            status = _dispatch_artifact(meta, args.overwrite, dataset, model_name)
            if status == "skip":
                skip += 1
            else:
                success += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            error += 1

    print(f"\n{'='*60}")
    print(f"Summary: {success} success, {skip} skip, {error} error")
    print("="*60)


if __name__ == "__main__":
    main()
