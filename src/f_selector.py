from boruta import BorutaPy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class BorutaFS(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.top_idx = None

    def fit(self, X, y):
        model = RandomForestClassifier(random_state=42, verbose=0, n_jobs=-1)

        self.selector = BorutaPy(
            model,
            random_state=42,
            verbose=0,
            early_stopping=True,
            max_iter=100,
            n_estimators=100,
        )
        self.selector.fit(X, y)
        self.top_idx = self.selector.support_

    def transform(self, X):
        assert self.top_idx is not None, "top_idx is None, call fit before transform"
        return X[:, self.top_idx]

    def fit_transform(self, X, y):  # type: ignore
        self.fit(X, y)
        return self.transform(X)
