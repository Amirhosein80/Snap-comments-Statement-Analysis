import re
import joblib
import argparse

from src.preprocessing import PersianTextProcessor
from src.config_reader import read_config
from src.logger import get_logger

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=r"config/config.yaml")
    parser.add_argument("--text", type=str, default="غذا بد بود.")
    return parser.parse_args()


class Pipeline:
    def __init__(self, model_path, tfidf_path) -> None:
        self.model = joblib.load(model_path)
        self.tfidf = joblib.load(tfidf_path)
        self.preprocessor = PersianTextProcessor()
        self.labels = ["Negative", "Positive"]

    def predict(self, text):
        processed_text = self.preprocessor.process_text(text)
        input_vector = self.tfidf.transform([processed_text]).toarray()
        prediction = self.model.predict(input_vector)
        label = self.labels[int(prediction[0])]
        return label
    
    def save(self, path):
        joblib.dump(self, path)

def main():
    args = parse_args()
    configs = read_config(args.config)["inference"]
    pipeline = Pipeline(configs["model_path"], configs["tfidf_path"])
    pipeline.save(configs["pipeline_path"])
    predict = pipeline.predict(args.text)
    
    logger.info(f"result: {predict}")
    

if  __name__ == "__main__":
    main()
