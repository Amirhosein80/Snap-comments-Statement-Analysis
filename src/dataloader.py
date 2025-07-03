from typing import Optional

import numpy as np
import pandas as pd


class DataLoader:
    def __init__(
        self,
        train_data_path: str,
        test_data_path: Optional[str] = None,
        drop_nan_labels: bool = True,
    ):

        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.train_dataset = pd.read_csv(train_data_path)
        self.train_dataset["labels"] = self.train_dataset["labels"].replace(-1, np.nan)
        if drop_nan_labels:
            self.train_dataset = self.train_dataset[
                self.train_dataset["labels"].notna()
            ]

        if test_data_path is not None:
            self.test_dataset = pd.read_csv(test_data_path)
            self.test_dataset["labels"] = self.test_dataset["labels"].replace(
                -1, np.nan
            )

            self.test_dataset = self.test_dataset[self.test_dataset["labels"].notna()]
        else:
            self.test_dataset = None

    def get_train_data(self):
        return (
            self.train_dataset.processed_comment.values,
            self.train_dataset.labels.values,
        )

    def get_test_data(self):
        if self.test_dataset is not None:
            return (
                self.test_dataset.processed_comment.values,
                self.test_dataset.labels.values,
            )
        else:
            return None, None
