import argparse
import sys
import json
import os
import re
import tempfile
from pathlib import Path

_HERE      = Path(__file__).resolve().parent    # external/elliot/
_PROJ_ROOT = _HERE.parent.parent               # d:/recsys-pipeline/
sys.path.insert(0, str(_HERE))


def _normalize_performance_files(perf_dir: Path, model_name: str) -> None:
    """Rename Elliot timestamp-named TSV/JSON output to <Model>_cutoff_<N>.[tsv|json].

    rec_cutoff_10_relthreshold_0_2026_08_09_11_20_09.tsv  →  VSM_cutoff_10.tsv
    bestmodelparams_cutoff_10_relthreshold_0_...json       →  bestmodelparams_VSM_cutoff_10.json
    """
    if not perf_dir.exists():
        return
    try:
        import pandas as pd
    except ImportError:
        print("  [WARN] pandas not available — skipping performance rename")
        return

    renamed = 0

    for tsv in sorted(perf_dir.glob("rec_cutoff_*.tsv"), key=lambda p: p.stat().st_mtime):
        m = re.search(r"rec_cutoff_(\d+)_", tsv.name)
        if not m:
            continue
        cutoff   = m.group(1)
        out_path = perf_dir / f"{model_name}_cutoff_{cutoff}.tsv"
        tsv.replace(out_path)
        renamed += 1
        print(f"  [eval] {tsv.name}  →  {out_path.name}")

    for jf in sorted(perf_dir.glob("bestmodelparams_cutoff_*.json"), key=lambda p: p.stat().st_mtime):
        m = re.search(r"cutoff_(\d+)_", jf.name)
        if not m:
            continue
        cutoff   = m.group(1)
        out_path = perf_dir / f"bestmodelparams_{model_name}_cutoff_{cutoff}.json"
        jf.replace(out_path)
        renamed += 1
        print(f"  [eval] {jf.name}  →  {out_path.name}")

    if renamed:
        print(f"  [eval] Normalized {renamed} file(s) in {perf_dir}")

from elliot.run import run_experiment

parser = argparse.ArgumentParser('Elliot evaluation')
parser.add_argument('-c', '--config', required=True, type=str, help='config file')
args = parser.parse_args()

try:
    import yaml
    with open(args.config, encoding='utf-8') as _f:
        _cfg = yaml.safe_load(_f)
except Exception as _e:
    sys.exit(f"ERROR: Failed to read config {args.config!r}: {_e}")

_proxy_cfg = (_cfg or {}).get('experiment', {}).get('models', {}).get('ProxyRecommender', {})

if isinstance(_proxy_cfg, dict) and _proxy_cfg.get('path') == 'selected_artifact':
    _source_model = _proxy_cfg.get('source_model')
    _dataset      = (_cfg.get('experiment') or {}).get('dataset')

    if not _source_model:
        sys.exit("[selected_artifact] ERROR: 'source_model' required in ProxyRecommender when path=selected_artifact")
    if not _dataset:
        sys.exit("[selected_artifact] ERROR: 'dataset' not found in experiment config")

    _meta_file = _PROJ_ROOT / "results" / "selected_artifacts" / _dataset / f"{_source_model}.json"
    if not _meta_file.exists():
        sys.exit(
            f"\n[selected_artifact] ERROR: No metadata for dataset={_dataset!r} model={_source_model!r}.\n"
            f"  Expected: {_meta_file}\n"
            f"  Run training first."
        )

    with open(_meta_file, encoding='utf-8') as _fh:
        _meta = json.load(_fh)

    _artifact_path = _meta.get('selected_artifact', '')
    if _artifact_path and not os.path.isabs(_artifact_path):
        _artifact_path = str(_PROJ_ROOT / _artifact_path)

    if not os.path.isfile(_artifact_path):
        sys.exit(
            f"\n[selected_artifact] ERROR: Artifact not found: {_artifact_path}\n"
            f"  Metadata (dataset={_dataset!r} model={_source_model!r}) points here.\n"
            f"  File may have been moved or deleted."
        )

    print("\n[selected_artifact]")
    print(f"  framework = {_meta.get('framework', '?')}")
    print(f"  dataset   = {_meta.get('dataset', '?')}")
    print(f"  model     = {_meta.get('model', '?')}")
    if 'selected_iteration' in _meta:
        print(f"  iteration = {_meta['selected_iteration']}")
    print(f"  path      = {_artifact_path}")

    # Inject resolved path, strip source_model (ProxyRecommender doesn't know it)
    _cfg['experiment']['models']['ProxyRecommender']['path'] = _artifact_path
    _cfg['experiment']['models']['ProxyRecommender'].pop('source_model', None)

    # Resolve performance output dir (relative to external/elliot/ CWD)
    _perf_raw = (_cfg.get('experiment') or {}).get('path_output_rec_performance', '')
    _perf_dir = Path(os.path.abspath(os.path.join(str(_PROJ_ROOT), _perf_raw))) if _perf_raw else None

    # Write temp YAML in same dir as original so relative paths resolve identically
    _orig_dir = os.path.dirname(os.path.abspath(args.config))
    _tmp_fd, _tmp_path = tempfile.mkstemp(suffix='.yml', dir=_orig_dir)
    try:
        with os.fdopen(_tmp_fd, 'w', encoding='utf-8') as _tf:
            yaml.dump(_cfg, _tf, default_flow_style=False, allow_unicode=True)
        run_experiment(_tmp_path)
    finally:
        try:
            os.unlink(_tmp_path)
        except Exception:
            pass

    if _perf_dir:
        _normalize_performance_files(_perf_dir, _source_model)
else:
    run_experiment(args.config)
    _perf_raw = (_cfg.get('experiment') or {}).get('path_output_rec_performance', '')
    if _perf_raw:
        _perf_dir = Path(os.path.abspath(os.path.join(str(_HERE), _perf_raw)))
        # best-effort model name: first key under models:
        _models = (_cfg.get('experiment') or {}).get('models', {})
        _model_name = next(iter(_models), 'model') if _models else 'model'
        _normalize_performance_files(_perf_dir, _model_name)
