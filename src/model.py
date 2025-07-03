import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.dataloader import DataLoader
from src.f_selector import BorutaFS
from src.logger import get_logger

logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, config):
        self.config = config["model_training"]

        self.supervised_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "feature_selector",
                    BorutaFS(),
                ),
                (
                    "model",
                    LinearSVC(
                        C=self.config["model"]["C"],
                        loss=self.config["model"]["loss"],
                        random_state=42,
                        verbose=0,
                    ),
                ),
            ]
        )

        self.supervised_dataloader = DataLoader(
            train_data_path=self.config["train_dataset"],
            test_data_path=self.config["test_dataset"],
            drop_nan_labels=True,
        )

        self.semi_supervised_training = False

        self.tfidf = TfidfVectorizer(
            ngram_range=(1, self.config["tfidf"]["ngram_max"]),
            max_df=self.config["tfidf"]["max_df"],
            min_df=self.config["tfidf"]["min_df"],
            max_features=self.config["tfidf"]["max_features"],
        )

        if self.config["semi_supervised"]:
            self.semi_supervised_pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "feature_selector",
                        BorutaFS(),
                    ),
                    (
                        "model",
                        LinearSVC(
                            C=self.config["model"]["C"],
                            loss=self.config["model"]["loss"],
                            random_state=42,
                            verbose=0,
                        ),
                    ),
                ]
            )

            self.semi_supervised_dataloader = DataLoader(
                train_data_path=self.config["train_dataset"],
                test_data_path=None,
                drop_nan_labels=False,
            )

            self.semi_supervised_training = True

    def save_model(self, path):
        joblib.dump(self.tfidf, os.path.join(path, "tfidf.pkl"))
        joblib.dump(
            self.supervised_pipeline, os.path.join(path, "supervised_pipeline.pkl")
        )

        if self.semi_supervised_training:
            joblib.dump(
                self.semi_supervised_pipeline,
                os.path.join(path, "semi_supervised_pipeline.pkl"),
            )

    def train_test_supervised(
        self, X_train, y_train, X_test, y_test
    ) -> tuple[dict, dict]:
        train_scores = cross_validate(
            self.supervised_pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=["accuracy", "f1", "precision", "recall"],
            verbose=0,
            n_jobs=-1,
        )

        self.supervised_pipeline.fit(X_train, y_train)

        start_time = time.time()
        y_pred = self.supervised_pipeline.predict(X_test)
        runtime = time.time() - start_time

        test_scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "runtime": runtime,
        }

        return train_scores, test_scores

    def train_test_semi_supervised(
        self, X_train_semi, y_train_semi, X_train_super, y_train_super, X_test, y_test
    ) -> tuple[dict, dict]:

        nan_labels = np.isnan(y_train_semi)

        self.supervised_pipeline.fit(X_train_super, y_train_super)
        y_train_semi[nan_labels] = self.supervised_pipeline.predict(
            X_train_semi[nan_labels]
        )

        train_scores = cross_validate(
            self.semi_supervised_pipeline,
            X_train_semi,
            y_train_semi,
            cv=5,
            scoring=["accuracy", "f1", "precision", "recall"],
            verbose=0,
            n_jobs=-1,
        )

        self.semi_supervised_pipeline.fit(X_train_semi, y_train_semi)

        start_time = time.time()
        y_pred = self.semi_supervised_pipeline.predict(X_test)
        runtime = time.time() - start_time

        test_scores = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "runtime": runtime,
        }

        return train_scores, test_scores

    def run(self):

        X_train_super, y_train_super = self.supervised_dataloader.get_train_data()
        X_test_super, y_test_super = self.supervised_dataloader.get_test_data()

        if self.semi_supervised_training:
            X_train_semi, y_train_semi = (
                self.semi_supervised_dataloader.get_train_data()
            )
            self.tfidf.fit(X_train_semi)
        else:
            self.tfidf.fit(X_train_super)

        logger.info("Model Training started")

        X_test_super = self.tfidf.transform(X_test_super).toarray()  # type: ignore
        X_train_super = self.tfidf.transform(X_train_super).toarray()  # type: ignore
        supervised_scores_valid, supervised_scores_test = self.train_test_supervised(
            X_train_super, y_train_super, X_test_super, y_test_super
        )
        # print(supervised_scores_valid.keys(), supervised_scores_test.keys())
        logger.info(
            f"Supervised Cross-validation accuracy: {supervised_scores_valid['test_accuracy'].mean()}"
        )
        logger.info(
            f"Supervised Cross-validation f1: {supervised_scores_valid['test_f1'].mean()}"
        )
        logger.info(
            f"Supervised Cross-validation precision: {supervised_scores_valid['test_precision'].mean()}"
        )
        logger.info(
            f"Supervised Cross-validation recall: {supervised_scores_valid['test_recall'].mean()}"
        )

        logger.info(f"Supervised Test accuracy: {supervised_scores_test['accuracy']}")
        logger.info(f"Supervised Test f1: {supervised_scores_test['f1']}")
        logger.info(f"Supervised Test precision: {supervised_scores_test['precision']}")
        logger.info(f"Supervised Test recall: {supervised_scores_test['recall']}")
        logger.info(f"Supervised Test runtime: {supervised_scores_test['runtime']}")

        print("\n\n")

        if self.semi_supervised_training:
            X_train_semi = self.tfidf.transform(X_train_semi).toarray()  # type: ignore
            semi_supervised_scores_valid, semi_supervised_scores_test = (
                self.train_test_semi_supervised(
                    X_train_semi,
                    y_train_semi,
                    X_train_super,
                    y_train_super,
                    X_test_super,
                    y_test_super,
                )
            )

            logger.info(
                f"Semi-supervised Cross-validation accuracy: {semi_supervised_scores_valid['test_accuracy'].mean()}"
            )
            logger.info(
                f"Semi-supervised Cross-validation f1: {semi_supervised_scores_valid['test_f1'].mean()}"
            )
            logger.info(
                f"Semi-supervised Cross-validation precision: {semi_supervised_scores_valid['test_precision'].mean()}"
            )
            logger.info(
                f"Semi-supervised Cross-validation recall: {semi_supervised_scores_valid['test_recall'].mean()}"
            )

            logger.info(
                f"Semi-supervised Test accuracy: {semi_supervised_scores_test['accuracy']}"
            )
            logger.info(f"Semi-supervised Test f1: {semi_supervised_scores_test['f1']}")
            logger.info(
                f"Semi-supervised Test precision: {semi_supervised_scores_test['precision']}"
            )
            logger.info(
                f"Semi-supervised Test recall: {semi_supervised_scores_test['recall']}"
            )
            logger.info(
                f"Semi-supervised Test runtime: {semi_supervised_scores_test['runtime']}"
            )

        self.save_model(self.config["save_path"])


if __name__ == "__main__":
    from src.config_reader import read_config

    configs = read_config("./config/config.yaml")
    model_training = ModelTraining(configs)
    model_training.run()
