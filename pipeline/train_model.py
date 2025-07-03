import argparse
from src.model import ModelTraining
from src.logger import get_logger
from src.config_reader import read_config

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Arguments for runing the code.
    
    Example
    ----------
    args = parse_args()
    config_path = args.config
    
    Returns
    -------
    return args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"config/config.yaml")
    return parser.parse_args()


def main():
    """
    Training the model.
    
    Example
    ----------
    >>> (in shell) python pipeline/train_model.py --config config/config.yaml

    """
    args = parse_args()
    configs = read_config(args.config)
    
    model = ModelTraining(config=configs)
    model.run()
    
if __name__ == "__main__":
    main()