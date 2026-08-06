import argparse
import sys
sys.path.append('../elliot')
from elliot.run import run_experiment
 
parser = argparse.ArgumentParser('Elliot evaluation')
parser.add_argument('-c','--config', default='bp_vsm.yml', type=str, help='config file')
 
args = parser.parse_args()
 
run_experiment(args.config)