import pandas as pd
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from src.label_funcs import LABEL_FUNCS


class DataLabeler:
    """
    Create labels with snorkel's weak supervision.
    """

    def __init__(self, data_dir: str, cardinality: int) -> None:
        """
        Parameters
        ----------
        data_dir: str
            The directory of the dataset to be labeled.
        cardinality: int
            The number of unique labels.

        Example
        ----------
        datalabeler = DataLabeler(data_dir, cardinality)
        datalabeler.create_labels()
        """
        self.data_dir = data_dir
        self.dataset = pd.read_csv(data_dir)
        self.applier = PandasLFApplier(LABEL_FUNCS)
        self.train_labels = self.applier.apply(self.dataset)
        self.label_model = LabelModel(cardinality=cardinality)

    def create_labels(self):
        self.label_model.fit(self.train_labels, n_epochs=100)  # type: ignore
        self.dataset["labels"] = self.label_model.predict(
            L=self.train_labels, tie_break_policy="abstain"  # type: ignore
        )
        self.dataset.to_csv(
            self.data_dir.replace("_processed", "_labeled"), index=False
        )


if __name__ == "__main__":
    train_datalabeler = DataLabeler("artifacts/train_processed.csv", 2)
    test_datalabeler = DataLabeler("artifacts/test_processed.csv", 2)
    train_datalabeler.create_labels()
    test_datalabeler.create_labels()

    print(train_datalabeler.train_labels)
