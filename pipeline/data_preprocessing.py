import os
import argparse

import test
from src.config_reader import read_config
from src.logger import get_logger
from src.preprocessing import DataPreprocessing
from src.create_labels import DataLabeler

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

def main() -> None:
    """
    Preprocessing and Labeling the dataset.
    
    Example
    ----------
    >>> (in shell) python pipeline/data_preprocessing.py --config config/config.yaml
    
    """
    logger.info("Starting data preprocessing")
    args = parse_args()
    configs = read_config(args.config)["datasets"]
    train_preoprocessor = DataPreprocessing(configs["train"],
                                            configs["artifact_dir"])
    test_preprocessor = DataPreprocessing(configs["test"],
                                          configs["artifact_dir"])
    
    
    train_preoprocessor.create_processed_csv()
    test_preprocessor.create_processed_csv()
    
    logger.info("Data preprocessing completed")
    logger.info("Starting data labeling")
    
    train_labelr = DataLabeler(os.path.join(configs["artifact_dir"], 
                                            "train_processed.csv"),
                               cardinality=configs["num_classes"])
    test_labelr = DataLabeler(os.path.join(configs["artifact_dir"], 
                                           "test_processed.csv"),
                               cardinality=configs["num_classes"])
    
    train_labelr.create_labels()
    test_labelr.create_labels()
    
    logger.info("Labels created")
    

if __name__ == "__main__":
    main()