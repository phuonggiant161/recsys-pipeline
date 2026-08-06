import argparse
import numpy as np
np.float = float

from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument(
    "-f", "--filetrained",
    required=True, type=str,
    help="path to the trained model file (checkpoint)"
)

args = parser.parse_args()

config, model, dataset, train_data, valid_data, test_data = \
    load_data_and_model(
        model_file=args.filetrained
    )

# adding additional metrics to the evaluation
config['metrics'].extend(['TailRecall', 'TailNDCG'])
config["tail_ratio"] = 2

trainer = Trainer(config, model)
trainer.eval_collector.data_collect(train_data)

result = trainer.evaluate(
    test_data,
    load_best_model=False,
    show_progress=config["show_progress"]
)

print(result)