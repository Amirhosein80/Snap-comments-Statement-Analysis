# SnapFood comments Statement Analysis

An ML model to predict whether a statement is a positive or negative statement based on the snapfood comments.

## Dataset 

The dataset is from an Statement Analysis task on the Quera College's Question bank [(link)](https://quera.org/problemset/125360).

This dataset dosn't contain any labels. So in order to create labels for this dataset, we used [Snorkel](https://snorkel.ai/data-labeling/) library with simple labeling functions.

## Installation
You can use install this package by running this command in your terminal. we highly recomend to use an virtual environment.

```cmd
pip install -e .
```

## Usage 
Run the inference python file in your terminal with the config file path and the persian text. We used [parsivar](https://github.com/ICTRC/Parsivar) for persian text preprocessing, then a a tfidf vectorizer is used to convert the text into a vector. The vector is then passed with Brouta alghorithm with Random Forest for feature selection. Finally we used an Linear SVC model to predict the sentiment.

```python
python pipeline\infrence.py --config=config\config.yaml --text="غدا خوشمزه بود"
```

## Preprocessing and Lableing
You can run the preprocessing and lableing python file in your terminal with the config file path. to create the label_dataset for you with snorkel. We use [parsivar](https://github.com/ICTRC/Parsivar) library for persian text preprocessing. Then we defined three lableing functions and run them with [Snorkel](https://snorkel.ai/data-labeling/) to label the dataset.

```python
python pipeline\data_preprocessing.py --config=config\config.yaml
```

## Training
For trainig the tfidf, brouta feature selector and Linear SVC model, run the following command in the terminal. You can change the parameters in the config file inside the config folder and use your own hyperparameters.

```python
python pipeline\train_model.py --config=config\config.yaml
```


# Results
We evaluated our model with both cross validation with 5 folds and the test set. Our model has achived 0.9897 accuracy on cross validation and 0.9833 on test set.

### Supervised Learning Results
we only used the labeled data for training.

|  Metrics  |  Cross Validation | Test set |
|:------:|:--------:|:--------:|
| Accuracy | 0.9850 | 0.9826 |
| Precision | 0.9884 | 0.9855 |
| Recall | 0.9891 | 0.9886 |
| F1 | 0.9888 | 0.9870 |


### Semi Supervised Learning Results
Firstly we used the labeled data for training. then we used the trained model to predict the unlabeled data. then we used both predicted and labeled data for training.

|  Metrics  |  Cross Validation | Test set |
|:------:|:--------:|:--------:|
| Accuracy | 0.9897 | 0.9833 |
| Precision | 0.9880 | 0.9848 |
| Recall | 0.9907 | 0.9903 |
| F1 | 0.9894 | 0.9876 |



# Warning
### We only evaluated the model with the labels which are created by the snorkel. So we can't guarantee that the model always retrun the correct result. For example, if you type "غذا سرد بود" in the terminal, the model will return "negative" label which is correct. but if you type "بستنی سرد بود" in the terminal, the model will return "negative" label which is incorrect
