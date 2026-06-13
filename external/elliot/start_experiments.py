from elliot.run import run_experiment
import argparse

parser = argparse.ArgumentParser(description="Run sample main.")
parser.add_argument('--config', type=str, default='sample_hello_world')
parser.add_argument('--config-dir', type=str, default='config_files',
                    help="Directory containing config files (default: config_files)")
args = parser.parse_args()

run_experiment(f"{args.config_dir}/{args.config}.yml")
