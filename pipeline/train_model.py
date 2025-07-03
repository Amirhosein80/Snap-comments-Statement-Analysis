import argparse
from src.model import ModelTraining
from src.logger import get_logger
from src.config_reader import read_config

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"config/config.yaml")
    return parser.parse_args()

def main():
    
    args = parse_args()
    configs = read_config(args.config)
    
    model = ModelTraining(config=configs)
    model.run()
    
