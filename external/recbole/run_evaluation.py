import argparse
import csv
import sys
from pathlib import Path
import numpy as np
np.float = float

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPTS_DIR  = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
from selected_artifact import read_artifact, print_artifact

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f", "--filetrained",
    default=None, type=str,
    help="path to the trained model checkpoint (overrides metadata auto-lookup)"
)
parser.add_argument(
    "--dataset", default=None, type=str,
    help="Dataset name for auto-lookup from results/selected_artifacts/<dataset>/<model>.json"
)
parser.add_argument(
    "--model", default=None, type=str,
    help="Model name for auto-lookup from results/selected_artifacts/<dataset>/<model>.json"
)

args = parser.parse_args()

# Resolve checkpoint: -f wins; otherwise auto-lookup from metadata JSON
if args.filetrained:
    checkpoint_path = args.filetrained
else:
    if not args.dataset or not args.model:
        parser.error(
            "Must provide either -f/--filetrained, or both --dataset and --model for metadata auto-lookup."
        )
    meta = read_artifact(args.dataset, args.model)
    print_artifact(meta)
    checkpoint_path = meta["_resolved_path"]

config, model, dataset, train_data, valid_data, test_data = \
    load_data_and_model(
        model_file=checkpoint_path
    )

# adding additional metrics to the evaluation
_pop_metrics = [
    'PopRecall_pop1',     'PopRecall_pop2_5',   'PopRecall_pop6_10',
    'PopRecall_pop11_20', 'PopRecall_pop20plus',
    'PopNDCG_pop1',       'PopNDCG_pop2_5',     'PopNDCG_pop6_10',
    'PopNDCG_pop11_20',   'PopNDCG_pop20plus',
]
_user_metrics = [
    'UserRecall_u1',     'UserRecall_u2_5',   'UserRecall_u6_10',
    'UserRecall_u11_20', 'UserRecall_u20plus',
    'UserNDCG_u1',       'UserNDCG_u2_5',     'UserNDCG_u6_10',
    'UserNDCG_u11_20',   'UserNDCG_u20plus',
]
config['metrics'].extend(_pop_metrics + _user_metrics)

trainer = Trainer(config, model)
trainer.eval_collector.data_collect(train_data)

result = trainer.evaluate(
    test_data,
    load_best_model=False,
    show_progress=config["show_progress"]
)

print(result)


def _save_evaluation_result(dataset_name, model_name, ckpt_path, test_result):
    out_dir  = _PROJECT_ROOT / "results" / "recbole" / dataset_name / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_evaluation.tsv"

    row = {"dataset": dataset_name, "model": model_name, "checkpoint": ckpt_path,
           **test_result}

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)

    rel = out_file.relative_to(_PROJECT_ROOT)
    print(f"\n[evaluation] Saved:\n  {rel}")


_save_evaluation_result(
    dataset_name   = config["dataset"],
    model_name     = config["model"],
    ckpt_path      = checkpoint_path,
    test_result    = result,
)