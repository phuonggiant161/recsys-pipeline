import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECBOLE_SRC = os.path.join(PROJECT_ROOT, "external", "recbole")
sys.path.insert(0, RECBOLE_SRC)

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

args = parser.parse_args()

# Dynamically build the path based on the dataset string
runtime_overrides = {
    'data_path':      os.path.join(PROJECT_ROOT, 'data', 'recbole'),
    'checkpoint_dir': os.path.join(PROJECT_ROOT, 'saved', args.dataset),
}

run_recbole(
    model='SASRec',
    dataset=args.dataset,
    config_file_list=[args.config],
    config_dict=runtime_overrides,
)
