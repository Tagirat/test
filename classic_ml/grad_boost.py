import numpy as np
from sklearn.tree import DecisionTreeRegressor


class GBCustomRegressor:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.init_value = None

    def fit(self, x, y):
        self.init_value = np.mean(y)
        predictions = np.full_like(y, self.init_value, dtype=np.float64)

        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(x, residuals)
            predictions += self.learning_rate * tree.predict(x)
            self._estimators.append(tree)

    def predict(self, x):
        predictions = np.full(x.shape[0], self.init_value, dtype=np.float64)
        for tree in self._estimators:
            predictions += self.learning_rate * tree.predict(x)

        return predictions

    @property
    def estimators_(self):
        return self._estimators

    @estimators_.setter
    def estimators_(self, value):
        self._estimators = value


class GBCustomClassifier:
    def __init__(
            self,
            *,
            learning_rate=0.1,
            n_estimators=100,
            criterion="friedman_mse",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=3,
            random_state=None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        self._estimators = []
        self.init_value = None

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        y = np.array(y, dtype=np.float64)
        pos_class_prob = np.mean(y)
        self.init_value = np.log(pos_class_prob / (1 - pos_class_prob))
        predictions = np.full_like(y, self.init_value, dtype=np.float64)

        for _ in range(self.n_estimators):
            probas = self._sigmoid(predictions)
            residuals = y - probas
            tree = DecisionTreeRegressor(
                criterion=self.criterion,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            tree.fit(x, residuals)
            leaf_preds = tree.apply(x)
            for leaf in np.unique(leaf_preds):
                mask = leaf_preds == leaf
                numerator = np.sum(residuals[mask])
                denominator = np.sum((y[mask] - residuals[mask]) * (1 - y[mask] + residuals[mask]))
                if denominator == 0:
                    tree_pred = 0
                else:
                    tree_pred = numerator / denominator
                tree.tree_.value[leaf, 0, 0] = tree_pred
            predictions += self.learning_rate * tree.predict(x)
            self._estimators.append(tree)

    def predict_proba(self, x):
        logits = np.full(x.shape[0], self.init_value, dtype=np.float64)
        for tree in self._estimators:
            logits += self.learning_rate * tree.predict(x)
        proba = self._sigmoid(logits)
        return np.vstack([1 - proba, proba]).T

    def predict(self, x):
        proba = self.predict_proba(x)
        return (proba[:, 1] > 0.5).astype(int)

    @property
    def estimators_(self):
        return self._estimators

    @estimators_.setter
    def estimators_(self, value):
        self._estimators = value
