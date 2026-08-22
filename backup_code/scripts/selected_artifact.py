"""
Shared utilities for reading/writing selected-artifact metadata.

Metadata is stored at:
    results/selected_artifacts/<dataset>/<model>.json

Rules enforced here:
- Write is atomic (rename from .tmp so the file is never half-written).
- Read fails fast with a clear message when JSON or artifact is missing.
- Never uses "latest file", "max iteration", or scan-to-guess logic.
"""
import json
from pathlib import Path

_PROJECT_ROOT  = Path(__file__).resolve().parent.parent
_ARTIFACTS_DIR = _PROJECT_ROOT / "results" / "selected_artifacts"


def write_artifact(*, framework, dataset, model, selected_artifact,
                   artifact_type, **extra):
    """Write (or atomically overwrite) selected-artifact metadata.

    Returns (meta_file_path, data_dict).
    """
    meta_dir = _ARTIFACTS_DIR / dataset
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_file = meta_dir / f"{model}.json"

    data = {
        "framework":         framework,
        "dataset":           dataset,
        "model":             model,
        "selected_artifact": str(selected_artifact),
        "artifact_type":     artifact_type,
    }
    data.update(extra)

    tmp = meta_file.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(meta_file)

    return meta_file, data


def read_artifact(dataset, model, *, resolve_path=True):
    """Read and validate selected-artifact metadata.

    If resolve_path=True (default), verifies the artifact file exists and
    adds '_resolved_path' (absolute str) to the returned dict.

    Raises FileNotFoundError with a descriptive message if:
    - The metadata JSON does not exist.
    - resolve_path=True and the artifact file does not exist.

    Never falls back to latest/max-iteration.
    """
    meta_file = _ARTIFACTS_DIR / dataset / f"{model}.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"\n[selected_artifact] No metadata found for "
            f"dataset={dataset!r} model={model!r}.\n"
            f"  Expected: {meta_file}\n"
            f"  Run training first so this file is generated."
        )

    with open(meta_file, encoding="utf-8") as fh:
        data = json.load(fh)

    if resolve_path:
        artifact_path = Path(data["selected_artifact"])
        if not artifact_path.is_absolute():
            artifact_path = _PROJECT_ROOT / artifact_path
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"\n[selected_artifact] Artifact file not found: {artifact_path}\n"
                f"  Metadata (dataset={dataset!r} model={model!r}) points here.\n"
                f"  The file may have been moved or deleted."
            )
        data["_resolved_path"] = str(artifact_path)

    return data


def print_artifact(data):
    """Print selected-artifact info in a consistent format."""
    print("\n[selected_artifact]")
    print(f"  framework = {data.get('framework', '?')}")
    print(f"  dataset   = {data.get('dataset', '?')}")
    print(f"  model     = {data.get('model', '?')}")
    if "selected_iteration" in data:
        print(f"  iteration = {data['selected_iteration']}")
    print(f"  path      = {data.get('selected_artifact', '?')}")
