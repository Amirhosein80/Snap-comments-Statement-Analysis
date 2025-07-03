import os
import re

import pandas as pd
import tqdm
from parsivar import Normalizer, Tokenizer

from src.logger import get_logger

logger = get_logger(__name__)


PERSIAN_STOP_WORDS = {
    "و",
    "در",
    "به",
    "از",
    "که",
    "را",
    "این",
    "با",
    "است",
    "را",
    "برای",
    "آن",
    "یک",
    "خود",
    "های",
    "یا",
    "هر",
    "تا",
    "کند",
    "بر",
    "نیز",
    "شد",
    "شدند",
    "باشد",
    "کرد",
    "شود",
}


class PersianTextProcessor:
    """
    This is a text processor for Persian language.
    """

    def __init__(self) -> None:
        """
        Example
        -------
        >>> processor = PersianTextProcessor()
        >>> processed_text = processor.process_text("متن ورودی")

        """
        self.normalizer = Normalizer(
            pinglish_conversion_needed=True, statistical_space_correction=True
        )
        self.tokenizer = Tokenizer()

    def process_text(self, text: str) -> str:
        """
        Process the input text and return the cleaned text.

        Parameters
        ----------
            text: str - The input Persian text to be processed.

        Returns
        -------
            str: The processed text as a single string.
        """
        text = self.correct_elongation(text)
        text = text.replace("\u200c", " ")
        text = self.normalizer.normalize(text)
        text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        text = re.sub(r"[ـ\r]", "", text)

        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "]+",
            flags=re.UNICODE,
        )
        text = emoji_pattern.sub(r"", text)

        text = re.sub(r"\d+", " ", text)

        text = re.sub(r"\s+", " ", text).strip()

        tokens = self.tokenizer.tokenize_words(text)

        tokens = [word for word in tokens if word not in PERSIAN_STOP_WORDS]
        tokens = [word.replace("\u200c", "_") for word in tokens]

        return " ".join(tokens)

    def correct_elongation(self, text: str):
        """
        Corrects elongated characters in the given text.

        Parameters
        ----------
            text: str - The input text with potential elongated characters.

        Returns
        -------
            str: The text with elongated characters corrected.

        Example
        -------
        >>> corrected_text = correct_elongation("عااااالییی")

        """
        return re.sub(r"(.)\1{2,}", r"\1", text)


class DataPreprocessing:
    """
    This class process the raw datasets to create the processed texts for training our model.
    """

    def __init__(self, data_file: str, out_dir: str):
        """
        Parameters
        ----------
            data_file: str - The path to the CSV file containing the dataset.

            out_dir: str - The directory where the processed CSV file will be saved.

        Example
        ----------
        >>> preprocessing = DataPreprocessing('data.csv', 'output_directory')
        >>> preprocessing.create_processed_csv()


        Returns
        ----------
            None
        """
        try:
            self.dataset = pd.read_csv(data_file)
            self.filename = os.path.basename(data_file).replace(".csv", "")
        except Exception as e:
            logger.error(f"Error reading the data file: {e}")
            raise

        self.dataset["is_english"] = self.dataset.comment.apply(self.is_english)
        self.dataset = self.dataset[~self.dataset.is_english]

        cols = list(self.dataset.columns)
        cols.remove("comment")

        self.dataset.drop(columns=cols, inplace=True, axis=1)
        self.dataset.reset_index(drop=True, inplace=True)

        self.processor = PersianTextProcessor()
        self.out_dir = out_dir

    def is_english(self, text):
        """
        Detect the given text is persian or english.

        Parameters
        ----------
            text: str - The input text to detect the language of.
        Returns
        ----------
            bool: False if the text is persian, True if the text is english.
        """
        persian_count = 0
        english_count = 0

        for char in text:
            if "\u0600" <= char <= "\u06ff":
                persian_count += 1

            elif ("A" <= char <= "Z") or ("a" <= char <= "z"):
                english_count += 1

        if persian_count > english_count:
            return False
        else:
            return True

    def create_processed_csv(self):
        """
        Creates a processed CSV file from the dataset.
        """
        processed_data = []
        loop = tqdm.tqdm(
            range(len(self.dataset.comment)), total=len(self.dataset.comment)
        )

        for index in loop:
            processed_text = self.processor.process_text(
                self.dataset.comment.iloc[index]
            )
            processed_data.append(processed_text)

        self.dataset["processed_comment"] = processed_data

        os.makedirs(self.out_dir, exist_ok=True)
        out_dir = os.path.join(self.out_dir, f"{self.filename}_processed.csv")
        self.dataset.to_csv(out_dir, index=False)
        logger.info(f"Processed CSV saved to {out_dir}")


if __name__ == "__main__":

    preprocessing = DataPreprocessing("data/test.csv", "artifacts")
    preprocessing.create_processed_csv()
    preprocessing = DataPreprocessing("data/train.csv", "artifacts")
    preprocessing.create_processed_csv()
