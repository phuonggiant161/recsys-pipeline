"""
Train a LibRecommender TwoTower model on the existing RecBole train/valid/test
splits, using global TF-IDF/SVD item content embeddings as dense item features.

Pipeline:
    RecBole train/valid/test .inter (user_id, item_id only)
    + global item content embeddings (data/processed/item_embeddings/*.parquet)
    -> LibRecommender DatasetFeat (item dense features)
    -> TwoTower, manual epoch loop with early stopping on validation NDCG@20
    -> reload BEST checkpoint (never the last epoch)
    -> Top-K recommendation TSV for TEST users (train+valid masked, test kept)
    -> register selected artifact (framework="librecommender")

Does NOT resplit data, does NOT rebuild embeddings, does NOT implement the
RecBole external evaluator (a later task will read the recommendation TSV
produced here).

Must be run with the LibRecommender environment, e.g.:
    .venv-twotower\\Scripts\\python.exe scripts\\run_two_tower.py --dataset hm_random_keep0.1

Usage:
    python scripts/run_two_tower.py --dataset hm_random_keep0.1
    python scripts/run_two_tower.py --dataset amazon_random_keep0.5 --overwrite
    python scripts/run_two_tower.py --all --overwrite
    python scripts/run_two_tower.py --all --filter amazon --overwrite
    python scripts/run_two_tower.py --all --temperature -0.1 --learning-rate 0.001 --overwrite
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from selected_artifact import write_artifact, print_artifact as _print_artifact  # noqa: E402

from libreco.algorithms import TwoTower  # noqa: E402
from libreco.data import DataInfo, DatasetFeat  # noqa: E402
from libreco.evaluation import evaluate  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

RECBOLE_DATA_ROOT = PROJECT_ROOT / "data" / "recbole"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "processed" / "item_embeddings"
RESULTS_ROOT = PROJECT_ROOT / "results" / "librecommender"

MODEL_NAME = "TwoTower"
HIDDEN_UNITS = (128, 64, 32)

_FAMILY_EMBEDDING_FILE = {
    "hm": "hm_tfidf_svd64.parquet",
    "amazon": "amazon_tfidf_svd64.parquet",
}


# ── Dataset family / path resolution ───────────────────────────────────────────

def detect_family(dataset: str) -> str:
    if dataset.startswith("hm"):
        return "hm"
    if dataset.startswith("amazon"):
        return "amazon"
    raise ValueError(
        f"Cannot detect dataset family from name {dataset!r} "
        f"(expected it to start with 'hm' or 'amazon'). "
        f"Pass --embedding-path explicitly to override auto-resolution."
    )


def resolve_embedding_path(dataset: str, override: Path | None) -> Path:
    if override is not None:
        return Path(override)
    family = detect_family(dataset)
    path = EMBEDDINGS_DIR / _FAMILY_EMBEDDING_FILE[family]
    if not path.exists():
        raise FileNotFoundError(
            f"Auto-resolved embedding file not found for family={family!r}: {path}\n"
            f"  Run scripts/build_two_tower_item_embeddings.py --dataset {family} first, "
            f"or pass --embedding-path explicitly."
        )
    return path


# ── Step: load RecBole .inter splits (user_id, item_id only) ─────────────────

def load_inter_split(path: Path) -> pd.DataFrame:
    """Read a RecBole atomic .inter file and return columns [user, item] as str.

    Only user_id:token and item_id:token are used -- timestamp/item_id_list are
    ignored, since Two-Tower here only needs static (user, item) interactions.
    RecBole token IDs are treated as strings throughout (see module docstring).
    """
    if not path.exists():
        raise FileNotFoundError(f"RecBole atomic file not found: {path}")
    df = pd.read_csv(
        path, sep="\t", dtype=str,
        usecols=["user_id:token", "item_id:token"],
    )
    df = df.rename(columns={"user_id:token": "user", "item_id:token": "item"})
    if df["user"].isna().any() or df["item"].isna().any():
        raise RuntimeError(f"Null user/item token found in {path} -- refusing to proceed.")
    return df.reset_index(drop=True)


# ── Step: split invariant validation ──────────────────────────────────────────

def _assert_subset(sub_name: str, sub_ids: set, sup_name: str, sup_ids: set) -> None:
    missing = sub_ids - sup_ids
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            f"Split invariant violated: {sub_name} is not a subset of {sup_name}. "
            f"{len(missing)} {sub_name} id(s) missing from {sup_name}. "
            f"Sample: {sample}"
        )


def validate_split_invariants(train_df, valid_df, test_df) -> None:
    train_users, valid_users, test_users = (
        set(train_df["user"]), set(valid_df["user"]), set(test_df["user"])
    )
    train_items, valid_items, test_items = (
        set(train_df["item"]), set(valid_df["item"]), set(test_df["item"])
    )
    _assert_subset("valid users", valid_users, "train users", train_users)
    _assert_subset("test users", test_users, "train users", train_users)
    _assert_subset("valid items", valid_items, "train items", train_items)
    _assert_subset("test items", test_items, "train items", train_items)
    print(
        f"  [OK] split invariants: valid/test users and items are subsets of train "
        f"(train users={len(train_users):,}, items={len(train_items):,})"
    )


# ── Step: item embedding coverage + attach ────────────────────────────────────

def load_item_embeddings(path: Path) -> tuple[pd.DataFrame, list[str]]:
    emb = pd.read_parquet(path)
    if "item_id" not in emb.columns:
        raise RuntimeError(f"Embedding file {path} is missing an 'item_id' column.")
    emb_cols = [c for c in emb.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError(f"Embedding file {path} has no emb_* columns.")
    # Preserve emb_1..emb_N numeric order regardless of column order on disk.
    emb_cols = sorted(emb_cols, key=lambda c: int(c.split("_", 1)[1]))

    null_mask = emb["item_id"].isna()
    n_null = int(null_mask.sum())
    if n_null > 0:
        sample_idx = emb.index[null_mask][:10].tolist()
        raise RuntimeError(
            f"{n_null} null item_id(s) found in embedding file {path} "
            f"(checked before str casting, since NaN would otherwise silently "
            f"become the string \"nan\"). Sample row indices: {sample_idx}"
        )
    emb["item_id"] = emb["item_id"].astype(str)
    n_dup = int(emb["item_id"].duplicated().sum())
    if n_dup > 0:
        raise RuntimeError(
            f"{n_dup} duplicate item_id(s) in embedding file {path} -- refusing to proceed."
        )
    values = emb[emb_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError(f"Embedding file {path} contains NaN/Inf values.")

    print(
        f"  [OK] item embeddings loaded: {path}\n"
        f"       items={len(emb):,}  dims={len(emb_cols)}"
    )
    return emb[["item_id"] + emb_cols], emb_cols


def validate_embedding_coverage(all_items: set, emb: pd.DataFrame) -> None:
    emb_items = set(emb["item_id"])
    missing = all_items - emb_items
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            f"Item embedding coverage gap: {len(missing)} item(s) in "
            f"train+valid+test have no embedding. Sample: {sample}\n"
            f"  This must not happen for the current experiment item catalog -- "
            f"investigate the embedding build (scripts/build_two_tower_item_embeddings.py) "
            f"rather than silently dropping interactions here."
        )
    print(f"  [OK] embedding coverage: all {len(all_items):,} train+valid+test items have an embedding")


def attach_embeddings(df: pd.DataFrame, emb: pd.DataFrame, emb_cols: list[str], label: str) -> pd.DataFrame:
    n_before = len(df)
    out = df.merge(emb.rename(columns={"item_id": "item"}), on="item", how="left")
    if len(out) != n_before:
        raise RuntimeError(
            f"BUG: row count changed while attaching embeddings to {label} "
            f"({n_before} -> {len(out)}). Merge must be one-to-one on item."
        )
    if out[emb_cols].isna().any().any():
        n_null = int(out[emb_cols].isna().any(axis=1).sum())
        raise RuntimeError(
            f"BUG: {n_null} row(s) in {label} have null embedding after merge "
            f"despite passing coverage validation."
        )
    out["label"] = 1.0
    # DatasetFeat requires 'user','item' to be the first two columns.
    ordered_cols = ["user", "item", "label"] + emb_cols
    return out[ordered_cols].reset_index(drop=True)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


# ── Dataset discovery (for --all) ──────────────────────────────────────────────

def _discover_datasets(filter_str: str | None) -> list[str]:
    """Return sorted dataset names under data/recbole/ that have train/valid/test .inter files."""
    if not RECBOLE_DATA_ROOT.exists():
        raise SystemExit(
            f"ERROR: {RECBOLE_DATA_ROOT} does not exist.\n"
            "  Run: python scripts/recbole_prepare_data.py --all --overwrite"
        )
    datasets = []
    for d in sorted(RECBOLE_DATA_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if filter_str and filter_str not in name:
            continue
        missing = [s for s in ["train", "valid", "test"] if not (d / f"{name}.{s}.inter").exists()]
        if missing:
            print(f"  [SKIP] dataset={name}  reason=missing inter files: {missing}")
        else:
            datasets.append(name)
    if not datasets:
        suffix = f" matching '{filter_str}'" if filter_str else ""
        raise SystemExit(f"No valid datasets found in {RECBOLE_DATA_ROOT}{suffix}")
    return datasets


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LibRecommender TwoTower on existing RecBole splits with TF-IDF/SVD item content embeddings."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Dataset variant name, e.g. hm_random_keep0.1")
    group.add_argument("--all", action="store_true", help="Run on all valid datasets in data/recbole/")
    p.add_argument("--filter", default=None, help="(with --all) only datasets whose name contains this substring")
    p.add_argument("--embedding-path", default=None, type=Path,
                    help="Override auto-resolved item embedding parquet path.")
    p.add_argument("--overwrite", action="store_true",
                    help="Retrain and overwrite existing TwoTower outputs for this dataset.")
    p.add_argument("--noprogressbar", action="store_true", help="Reduce fit() print verbosity.")
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=0.0,
                    help="Minimum NDCG@20 improvement to reset patience (default 0 = strict >).")
    p.add_argument("--embed-size", type=int, default=128, help="TwoTower tower output embedding size.")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--temperature", type=float, default=0.1,
                    help="Softmax loss temperature (logits are divided by this before softmax). "
                         "<=0 treats it as a learnable variable, trained jointly with the model.")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument("--rec-batch-size", type=int, default=256,
                    help="Users per recommend_user() batch call during Top-K generation.")
    return p.parse_args()


# ── Single-dataset run ──────────────────────────────────────────────────────────

def run_one(dataset: str, args: argparse.Namespace) -> str:
    """Train + evaluate TwoTower for one dataset. Returns 'ok' or 'skip'."""
    model_dir = RESULTS_ROOT / dataset / "model" / MODEL_NAME
    training_dir = RESULTS_ROOT / dataset / "training"
    recs_dir = RESULTS_ROOT / dataset / "recs"
    history_path = training_dir / f"{MODEL_NAME}_history.tsv"
    meta_path = training_dir / f"{MODEL_NAME}_meta.json"
    recs_path = recs_dir / f"{MODEL_NAME}.tsv"

    if recs_path.exists() and not args.overwrite:
        print(f"[SKIP] {recs_path} already exists (use --overwrite to retrain).")
        return "skip"

    print(f"\n[RUN TWO-TOWER] dataset={dataset}")
    print(
        f"  hyperparameters: max_epochs={args.max_epochs} patience={args.patience} "
        f"min_delta={args.min_delta} embed_size={args.embed_size} batch_size={args.batch_size} "
        f"lr={args.learning_rate} temperature={args.temperature} topk={args.topk} seed={args.seed} "
        f"hidden_units={HIDDEN_UNITS} loss_type=softmax norm_embed=True"
    )
    set_all_seeds(args.seed)

    # ── Load RecBole splits (unchanged membership) ───────────────────────────
    inter_dir = RECBOLE_DATA_ROOT / dataset
    train_df = load_inter_split(inter_dir / f"{dataset}.train.inter")
    valid_df = load_inter_split(inter_dir / f"{dataset}.valid.inter")
    test_df = load_inter_split(inter_dir / f"{dataset}.test.inter")
    print(
        f"  Loaded .inter: train={len(train_df):,} valid={len(valid_df):,} test={len(test_df):,}"
    )

    validate_split_invariants(train_df, valid_df, test_df)

    # ── Item embeddings: resolve, load, validate coverage, attach ───────────
    embedding_path = resolve_embedding_path(dataset, args.embedding_path)
    emb, emb_cols = load_item_embeddings(embedding_path)
    if len(emb_cols) != 64:
        print(f"  [INFO] embedding file has {len(emb_cols)} dims (not the usual 64) -- using as-is.")

    all_items = set(train_df["item"]) | set(valid_df["item"]) | set(test_df["item"])
    validate_embedding_coverage(all_items, emb)

    train_feat = attach_embeddings(train_df, emb, emb_cols, "train")
    valid_feat = attach_embeddings(valid_df, emb, emb_cols, "valid")
    test_feat = attach_embeddings(test_df, emb, emb_cols, "test")

    # Keep the raw (pre-embedding) user/item sets for masking + validation --
    # embeddings are per-item and don't affect membership, but we use the
    # smaller frames for set operations below for clarity.
    train_items_by_user: dict[str, set] = train_df.groupby("user")["item"].apply(set).to_dict()
    valid_items_by_user: dict[str, set] = valid_df.groupby("user")["item"].apply(set).to_dict()
    test_users = set(test_df["user"])

    # ── Build DatasetFeat: item dense features only, no user/sparse feats ───
    DatasetFeat.train_called = False  # defensive reset in case of prior calls in-process
    train_data, data_info = DatasetFeat.build_trainset(
        train_data=train_feat,
        user_col=None,
        item_col=emb_cols,
        sparse_col=None,
        dense_col=emb_cols,
        # Item content features are identical across all interaction rows of
        # the same item (a per-item join), so picking any single row's
        # feature vector per item is correct regardless of sort stability --
        # see feature/unique.py::_compress_unique_values (both branches
        # always compress to exactly one row per item; unique_feat only
        # changes the sort algorithm used, quicksort vs mergesort, which
        # only matters for tie-breaking among *different* feature values per
        # item, not present here). Safe speed optimization, not a
        # correctness change.
        unique_feat=True,
        seed=args.seed,
    )
    valid_data = DatasetFeat.build_evalset(valid_feat, shuffle=False, seed=args.seed)
    test_data = DatasetFeat.build_testset(test_feat, shuffle=False, seed=args.seed)

    print(
        f"  DatasetFeat built: n_users={data_info.n_users:,} n_items={data_info.n_items:,} "
        f"train_interactions={len(train_data):,} valid_interactions={len(valid_data)} "
        f"test_interactions={len(test_data)} dense_feature_count={len(emb_cols)}"
    )

    # ── Build model ONCE; outer loop manages epochs manually ─────────────────
    tf.compat.v1.reset_default_graph()
    model = TwoTower(
        task="ranking",
        data_info=data_info,
        loss_type="softmax",
        embed_size=args.embed_size,
        n_epochs=1,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        hidden_units=HIDDEN_UNITS,
        norm_embed=True,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(f"  norm_embed={model.norm_embed}  temperature={args.temperature}"
          f"{' (learned, <=0)' if args.temperature <= 0 else ''}")

    fit_verbose = 0 if args.noprogressbar else 1

    # ── Manual epoch loop with early stopping on validation NDCG@20 ──────────
    best_ndcg = float("-inf")
    best_epoch = None
    no_improve = 0
    history_rows: list[dict] = []
    stopped_reason = None
    prev_emb_norms = None  # (user_norm, item_norm) from the previous epoch, for the staleness check below

    model_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    for epoch in range(1, args.max_epochs + 1):
        # Invalidate the cached NumPy inference embeddings before every
        # outer-loop fit() call. LibRecommender's EmbedBase.fit() (which
        # TwoTower.fit() delegates to via super().fit()) only calls
        # self.set_embeddings() -- the step that reads the CURRENT trained TF
        # weights into user_embeds_np/item_embeds_np -- when
        # self.user_embeds_np is None:
        #     if self.user_embeds_np is None:
        #         self.set_embeddings()
        # That's True only on the very first fit() call; every later call
        # skips it. The trainer's own internal set_embeddings() refresh
        # (training/tf_trainer.py) is separately gated behind `verbose > 1`,
        # which we never pass (fit_verbose is 0 or 1) -- so nothing else
        # refreshes it either. Meanwhile TF training itself continues
        # correctly across fit() calls (model_built/trainer are reused, the
        # graph is never rebuilt) -- only the NumPy inference cache used by
        # recommend_user()/predict()/evaluate() was going stale. Confirmed
        # empirically: without this reset, validation NDCG@20 was frozen at
        # the epoch-1 value for 11 straight epochs on hm_random_keep0.1.
        model.user_embeds_np = None
        model.item_embeds_np = None

        model.fit(
            train_data,
            neg_sampling=True,  # implicit positive-only ranking data
            verbose=fit_verbose,
            shuffle=True,
        )

        # Defensive sanity check: confirm the inference embeddings were
        # actually rebuilt for this epoch (cheap -- two L2 norms of arrays
        # already in memory, no large dumps, negligible overhead). A repeated
        # *NDCG* value across epochs can legitimately happen (e.g. a genuine
        # plateau or a top-10 tie), so that alone is never treated as a
        # failure; this only warns when the embedding cache itself looks
        # like it did not change at all, which is the actual symptom of the
        # bug this loop now guards against.
        cur_emb_norms = (
            float(np.linalg.norm(model.user_embeds_np)),
            float(np.linalg.norm(model.item_embeds_np)),
        )
        if prev_emb_norms is not None and cur_emb_norms == prev_emb_norms:
            print(
                f"  [WARN][epoch {epoch}] user/item embedding norms identical to the "
                f"previous epoch ({cur_emb_norms[0]:.6f}, {cur_emb_norms[1]:.6f}) -- "
                f"the inference embedding cache may not be refreshing."
            )
        prev_emb_norms = cur_emb_norms

        eval_result = evaluate(
            model=model,
            data=valid_data,
            neg_sampling=True,  # required for positive-only labels; unused by listwise ndcg itself
            metrics=["ndcg"],
            k=20,
            sample_user_num=None,  # ALL validation users, never a sample
            seed=args.seed,
        )
        valid_ndcg = float(eval_result["ndcg"])

        is_best = valid_ndcg > best_ndcg + args.min_delta
        if is_best:
            best_ndcg = valid_ndcg
            best_epoch = epoch
            no_improve = 0
            data_info.save(path=str(model_dir), model_name=MODEL_NAME)
            model.save(str(model_dir), MODEL_NAME, manual=True, inference_only=True)
        else:
            no_improve += 1

        history_rows.append({
            "epoch": epoch,
            "valid_ndcg20": round(valid_ndcg, 6),
            "is_best": int(is_best),
            "no_improve": no_improve,
        })
        print(
            f"  [epoch {epoch:3d}] valid_ndcg@20={valid_ndcg:.6f}  "
            f"best={best_ndcg:.6f} (epoch {best_epoch})  "
            f"no_improve={no_improve}/{args.patience}  "
            f"emb_norm(user={cur_emb_norms[0]:.4f}, item={cur_emb_norms[1]:.4f})"
        )

        if no_improve >= args.patience:
            stopped_reason = f"no_improve_patience_reached({args.patience})"
            break
    else:
        stopped_reason = f"max_epochs_reached({args.max_epochs})"

    elapsed = time.time() - t_start
    print(f"  Training finished: best_epoch={best_epoch} best_valid_ndcg20={best_ndcg:.6f} "
          f"stopped_reason={stopped_reason} elapsed={elapsed:.1f}s")

    pd.DataFrame(history_rows, columns=["epoch", "valid_ndcg20", "is_best", "no_improve"]).to_csv(
        history_path, sep="\t", index=False
    )
    print(f"  Training history: {history_path}")

    # ── Reload BEST checkpoint (never the in-memory last-epoch model) ───────
    del model
    tf.compat.v1.reset_default_graph()
    loaded_data_info = DataInfo.load(path=str(model_dir), model_name=MODEL_NAME)
    best_model = TwoTower.load(str(model_dir), MODEL_NAME, loaded_data_info, manual=True)
    print(f"  Reloaded best checkpoint from {model_dir} (epoch {best_epoch})")

    # Basic sanity prediction/recommendation after reload.
    sanity_user = next(iter(test_users))
    sanity_recs = best_model.recommend_user(
        user=sanity_user, n_rec=5, filter_consumed=True, inner_id=False, random_rec=False
    )
    sanity_items = list(sanity_recs[sanity_user])
    sanity_scores = best_model.predict(
        user=[sanity_user] * len(sanity_items), item=sanity_items, inner_id=False
    )
    assert len(sanity_items) > 0, "sanity check failed: reloaded model returned 0 recommendations"
    assert np.isfinite(np.asarray(sanity_scores)).all(), "sanity check failed: non-finite predict() scores"
    print(f"  [OK] sanity check on reloaded model: user={sanity_user} -> {len(sanity_items)} recs, scores finite")

    # ── Optional reference-only NDCG@20 on TEST (NOT the thesis final metric) ─
    ref_eval = evaluate(
        model=best_model, data=test_data, neg_sampling=True,
        metrics=["ndcg"], k=20, sample_user_num=None, seed=args.seed,
    )
    reference_test_ndcg20 = float(ref_eval["ndcg"])
    print(
        f"  [REFERENCE ONLY] LibRecommender test ndcg@20={reference_test_ndcg20:.6f} "
        f"-- NOT the thesis final result; final evaluation will use the RecBole "
        f"external recommendation evaluator on the Top-K TSV produced below."
    )

    meta = {
        "dataset": dataset,
        "model": MODEL_NAME,
        "best_epoch": best_epoch,
        "best_valid_ndcg20": round(best_ndcg, 6),
        "stopped_reason": stopped_reason,
        "reference_only_test_ndcg20": round(reference_test_ndcg20, 6),
        "reference_only_note": "NOT the thesis final metric -- final evaluation uses the RecBole external evaluator on the recommendation TSV.",
        "hyperparameters": {
            "max_epochs": args.max_epochs, "patience": args.patience, "min_delta": args.min_delta,
            "embed_size": args.embed_size, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "temperature": args.temperature,
            "topk": args.topk, "seed": args.seed,
            "hidden_units": list(HIDDEN_UNITS), "loss_type": "softmax",
            "norm_embed": bool(best_model.norm_embed),
            "learned_temperature": (
                float(best_model.sess.run(best_model.temperature_var))
                if hasattr(best_model, "temperature_var") else None
            ),
        },
        "embedding_path": str(embedding_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  Training meta: {meta_path}")

    # ── Generate Top-K recommendations for TEST users only ───────────────────
    unknown_test_users = test_users - set(loaded_data_info.user2id.keys())
    if unknown_test_users:
        raise RuntimeError(
            f"BUG: {len(unknown_test_users)} test user(s) unknown to data_info despite "
            f"passing split invariant validation. Sample: {sorted(unknown_test_users)[:10]}"
        )

    catalog_items = set(train_df["item"])
    max_valid_per_user = max((len(v) for v in valid_items_by_user.values()), default=0)
    oversample_k = min(args.topk + max_valid_per_user, loaded_data_info.n_items)
    print(
        f"  Generating Top-{args.topk} recs for {len(test_users):,} test users "
        f"(oversample_k={oversample_k}, max_valid_per_user={max_valid_per_user})..."
    )

    rows: list[tuple] = []
    short_users: list[tuple] = []
    users_list = sorted(test_users)
    t_rec = time.time()
    for start in range(0, len(users_list), args.rec_batch_size):
        batch = users_list[start:start + args.rec_batch_size]
        # filter_consumed=True masks TRAIN-consumed items via data_info.user_consumed,
        # which was built exclusively from train_data in DatasetFeat.build_trainset --
        # this is the TRAIN masking mechanism. VALID is not known to data_info at all,
        # so it is masked explicitly below.
        batch_recs = best_model.recommend_user(
            user=batch, n_rec=oversample_k, filter_consumed=True,
            inner_id=False, random_rec=False,
        )
        for u in batch:
            candidates = list(batch_recs[u])
            valid_set = valid_items_by_user.get(u, set())
            kept = [it for it in candidates if it not in valid_set][: args.topk]
            if len(kept) < args.topk:
                short_users.append((u, len(kept)))
            if not kept:
                continue
            scores = best_model.predict(
                user=[u] * len(kept), item=kept, inner_id=False
            )
            scores = np.asarray(scores, dtype=np.float64)
            order = np.argsort(-scores)
            for idx in order:
                rows.append((u, kept[idx], float(scores[idx])))

    rec_elapsed = time.time() - t_rec
    print(f"  Recommendation generation done in {rec_elapsed:.1f}s -- {len(rows):,} rows")
    if short_users:
        print(
            f"  [WARN] {len(short_users)} test user(s) got fewer than topk={args.topk} "
            f"recommendations after train+valid masking (candidate pool too small). "
            f"Sample (user, n_kept): {short_users[:10]}"
        )

    rec_df = pd.DataFrame(rows, columns=["user_id", "item_id", "score"])

    # ── Validate recommendation TSV before writing ────────────────────────────
    _validate_recommendations(
        rec_df, test_users, train_items_by_user, valid_items_by_user,
        args.topk, catalog_items, short_users,
    )

    recs_dir.mkdir(parents=True, exist_ok=True)
    rec_df.to_csv(recs_path, sep="\t", index=False, header=False)
    print(f"  Recommendations: {recs_path}")

    # ── Register selected artifact ────────────────────────────────────────────
    _, artifact_meta = write_artifact(
        framework="librecommender",
        dataset=dataset,
        model=MODEL_NAME,
        selected_artifact=str(recs_path),
        artifact_type="recommendations",
        selected_iteration=best_epoch,
        best_valid_ndcg20=round(best_ndcg, 6),
    )
    _print_artifact(artifact_meta)

    # ── Release TF session/graph before the next dataset (--all loop) ───────
    try:
        best_model.sess.close()
    except Exception:
        pass
    del best_model
    tf.compat.v1.reset_default_graph()

    return "ok"


def _validate_recommendations(
    rec_df: pd.DataFrame,
    test_users: set,
    train_items_by_user: dict,
    valid_items_by_user: dict,
    topk: int,
    catalog_items: set,
    short_users: list,
) -> None:
    short_user_set = {u for u, _ in short_users}

    if not np.isfinite(rec_df["score"].to_numpy()).all():
        raise RuntimeError("Recommendation validation failed: non-finite score(s) found.")

    out_users = set(rec_df["user_id"])
    missing_users = (test_users - short_user_set) - out_users
    all_missing = test_users - out_users
    # Users with zero kept candidates are the explicitly-reported exception;
    # any other missing user is a silent-omission bug.
    unexpected_missing = all_missing - {u for u, n in short_users if n == 0}
    if unexpected_missing:
        raise RuntimeError(
            f"Recommendation validation failed: {len(unexpected_missing)} test user(s) "
            f"silently missing from output. Sample: {sorted(unexpected_missing)[:10]}"
        )
    extra_users = out_users - test_users
    if extra_users:
        raise RuntimeError(
            f"Recommendation validation failed: {len(extra_users)} output user(s) not in "
            f"test set. Sample: {sorted(extra_users)[:10]}"
        )

    bad_items = set(rec_df["item_id"]) - catalog_items
    if bad_items:
        raise RuntimeError(
            f"Recommendation validation failed: {len(bad_items)} recommended item(s) "
            f"outside the item catalog. Sample: {sorted(bad_items)[:10]}"
        )

    for u, group in rec_df.groupby("user_id"):
        items = group["item_id"].tolist()
        scores = group["score"].tolist()
        if len(items) != len(set(items)):
            raise RuntimeError(f"Recommendation validation failed: duplicate items for user {u}.")
        if len(items) > topk:
            raise RuntimeError(
                f"Recommendation validation failed: user {u} has {len(items)} recs > topk={topk}."
            )
        if u not in short_user_set and len(items) != topk:
            raise RuntimeError(
                f"Recommendation validation failed: user {u} has {len(items)} recs "
                f"(expected exactly {topk}, and was not reported as a short-candidate user)."
            )
        train_set = train_items_by_user.get(u, set())
        valid_set = valid_items_by_user.get(u, set())
        leaked_train = train_set.intersection(items)
        if leaked_train:
            raise RuntimeError(
                f"Recommendation validation failed: TRAIN item(s) leaked for user {u}: "
                f"{sorted(leaked_train)[:5]}"
            )
        leaked_valid = valid_set.intersection(items)
        if leaked_valid:
            raise RuntimeError(
                f"Recommendation validation failed: VALID item(s) leaked for user {u}: "
                f"{sorted(leaked_valid)[:5]}"
            )
        if scores != sorted(scores, reverse=True):
            raise RuntimeError(
                f"Recommendation validation failed: scores for user {u} are not descending."
            )

    print(
        f"  [OK] recommendation validation passed: {rec_df['user_id'].nunique():,} users, "
        f"{len(rec_df):,} rows, no train/valid leakage, no duplicates, "
        f"descending scores, all items in catalog."
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    datasets = [args.dataset] if args.dataset else _discover_datasets(args.filter)

    success = skip = error = 0
    errors: list[tuple[str, str]] = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"dataset={dataset}")
        print("="*60)
        try:
            status = run_one(dataset, args)
            if status == "skip":
                skip += 1
            else:
                success += 1
        except Exception as e:
            msg = str(e)
            print(f"  [ERROR] {msg}")
            errors.append((dataset, msg))
            error += 1

    if len(datasets) > 1:
        print(f"\n{'='*60}")
        print(f"Summary: {success} success, {skip} skip, {error} error")
        if errors:
            for ds, msg in errors:
                print(f"  FAIL  dataset={ds}")
                print(f"        {msg.splitlines()[0][:120]}")
        print("="*60)


if __name__ == "__main__":
    main()
