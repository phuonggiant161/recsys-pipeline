import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECBOLE_SRC = os.path.join(PROJECT_ROOT, "external", "recbole")
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# external/recbole must be first in sys.path to shadow any installed RecBole
sys.path.insert(0, RECBOLE_SRC)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from selected_artifact import write_artifact, print_artifact as _print_artifact

from recbole.quick_start import run_recbole

parser = argparse.ArgumentParser()

parser.add_argument(
    "-d", "--dataset",
    default='hm_random_keep0.9', type=str,
    help="dataset"
)

parser.add_argument(
    "-c", "--config",
    default=os.path.join(PROJECT_ROOT, "configs", "recbole", "sasrec.yaml"), type=str,
    help="config file"
)

parser.add_argument(
    "--noprogressbar",
    action="store_true",
    help="Turn off progress bar"
)

args = parser.parse_args()

# Dynamically build the path based on the dataset string
runtime_overrides = {
    "data_path":      os.path.join(PROJECT_ROOT, "data", "recbole"),
    "checkpoint_dir": os.path.join(PROJECT_ROOT, "saved", args.dataset),
    "show_progress":  not args.noprogressbar,
}

result = run_recbole(
    model='SASRec',
    dataset=args.dataset,
    config_file_list=[args.config],
    config_dict=runtime_overrides,
    saved=True,
)

# Register the checkpoint produced by THIS run as the selected artifact for
# (dataset, SASRec), mirroring scripts/recbole_run.py. Never scan the
# checkpoint directory for a "latest" .pth -- only the checkpoint_path
# returned by this training invocation is trusted.
if result is None or "checkpoint_path" not in result:
    raise RuntimeError(
        "run_recbole() returned no usable result for SASRec "
        f"(dataset={args.dataset!r}) -- refusing to write a selected-artifact "
        "record."
    )

checkpoint_path = result.get("checkpoint_path")
if not checkpoint_path:
    raise RuntimeError(
        "run_recbole() training succeeded but returned no checkpoint_path "
        f"(dataset={args.dataset!r}, saved=True) -- refusing to write a "
        "selected-artifact record without a real checkpoint."
    )
if not os.path.exists(checkpoint_path):
    raise RuntimeError(
        f"run_recbole() returned checkpoint_path={checkpoint_path!r} but the "
        "file does not exist on disk -- refusing to register it as the "
        "selected artifact."
    )

_, meta = write_artifact(
    framework="recbole",
    dataset=args.dataset,
    model="SASRec",
    selected_artifact=checkpoint_path,
    artifact_type="checkpoint",
)
_print_artifact(meta)
