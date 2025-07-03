import re
from tkinter import NO
import joblib
import argparse

from src.preprocessing import PersianTextProcessor
from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    """
    Arguments for runing the inference code.
    
    Example
    ----------
    args = parse_args()
    config_path = args.config
    text = args.text
    
    Returns
    ----------
    return args
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"config/config.yaml")
    parser.add_argument("--text", type=str, default="غذا بد بود.")
    return parser.parse_args()


class Pipeline:
    """
    Inference pipeline.
    """
    def __init__(self, model_path: str, tfidf_path: str) -> None:
        """
        Parameters
        ----------
        model_path: str
            Path to the model.
        tfidf_path: str
            Path to the tfidf model.
            
        Example
        ----------
        >>> pipeline = Pipeline(model_path="model.joblib", tfidf_path="tfidf.joblib")
        >>> result = pipeline.predict("some text")
        >>> pipeline.save("Path to save the pipeline")
        
        """
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        self.preprocessor = PersianTextProcessor()
        self.labels = ["Negative", "Positive"]

    def predict(self, text: str) -> str:
        """
        Predicet the statement.
        
        Parameters
        ----------
        text : str
            The statement to be predicted.
            
        Returns
        ----------
        str
            The predicted label.
            
        Examples
        ----------
        >>> result = pipeline.predict("some text")

        """
        processed_text = self.preprocessor.process_text(text)
        input_vector = self.tfidf.transform([processed_text]).toarray()
        prediction = self.model.predict(input_vector)
        label = self.labels[int(prediction[0])]
        return label
    
    def save(self, path: str) -> None:
        """
        Save the pipeline to a file.
        
        Parameters
        ----------
        path : str
            Path to save the pipeline.
        
        Example
        ----------
        >>> pipeline.save("Path to save the pipeline")
        
        """
        joblib.dump(self, path)

def main():
    """
    Inference for an text.
    
    Example
    ----------
    >>> (in shell) python pipeline/inference.py --config config/config.yaml --text "Your text"
    
    """
    args = parse_args()
    configs = read_config(args.config)["inference"]
    pipeline = Pipeline(configs["model_path"], configs["tfidf_path"])
    pipeline.save(configs["pipeline_path"])
    predict = pipeline.predict(args.text)
    
    logger.info(f"result: {predict}")
    

if  __name__ == "__main__":
    main()
