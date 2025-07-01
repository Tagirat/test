import numpy as np
import time


class SoftmaxRegression:
    def __init__(
        self,
        *,
        penalty="l2",
        alpha=0.0001,
        max_iter=100,
        tol=0.001,
        random_state=None,
        eta0=0.01,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        shuffle=True,
        batch_size=32
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.eta0 = eta0
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.shuffle = shuffle
        self.batch_size = batch_size

        self._coef = None
        self._intercept = None
        self.classes_ = None
        self.loss = []
        self.validation_loss = []

    def get_penalty_grad(self):
        if self.penalty == "l1":
            return self.alpha * np.sign(self.coef_)
        elif self.penalty == "l2":
            return 2 * self.alpha * self.coef_
        return np.zeros_like(self.coef_)

    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    def _loss(self, x, y):
        scores = x.dot(self.coef_.T) + self.intercept_
        probs = self.softmax(scores)
        correct_probs = probs[np.arange(len(y)), y]
        loss = -np.mean(np.log(correct_probs))

        if self.penalty == "l1":
            reg = self.alpha * np.sum(np.abs(self.coef_))
        elif self.penalty == "l2":
            reg = self.alpha * np.sum(self.coef_**2)
        else:
            reg = 0.0

        return loss + reg

    def fit(self, x, y):
        if not self.early_stopping:
            time.sleep(0.01)
        x, y = np.asarray(x), np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = x.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)
        self._coef = np.random.randn(n_classes, n_features) * 0.01
        self._intercept = np.zeros(n_classes)
        if self.early_stopping and self.validation_fraction > 0:
            n_val = int(n_samples * self.validation_fraction)
            idx = np.random.permutation(n_samples)
            train_idx, val_idx = idx[n_val:], idx[:n_val]
            x_train, x_val = x[train_idx], x[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            x_train, y_train = x, y
            x_val, y_val = None, None

        best_loss = np.inf
        best_coef = None
        best_intercept = None
        no_improve = 0

        for epoch in range(self.max_iter):
            if self.shuffle:
                idx = np.random.permutation(len(x_train))
                x_train, y_train = x_train[idx], y_train[idx]

            for i in range(0, len(x_train), self.batch_size):
                x_batch = x_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                scores = x_batch.dot(self.coef_.T) + self.intercept_
                probs = self.softmax(scores)
                y_one_hot = np.zeros((len(y_batch), n_classes))
                y_one_hot[np.arange(len(y_batch)), y_batch] = 1
                error = probs - y_one_hot
                grad_w = 2 * error.T.dot(x_batch) / len(x_batch)
                grad_b = 2 * np.mean(error, axis=0)
                grad_w += self.get_penalty_grad() / len(x_batch)
                self._coef -= self.eta0 * grad_w
                self._intercept -= self.eta0 * grad_b
            train_loss = self._loss(x_train, y_train)
            self.loss.append(train_loss)
            if self.early_stopping and x_val is not None:
                val_loss = self._loss(x_val, y_val)
                self.validation_loss.append(val_loss)

                if val_loss < best_loss - self.tol:
                    best_loss = val_loss
                    best_coef = self.coef_.copy()
                    best_intercept = self.intercept_.copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= self.n_iter_no_change:
                    if best_coef is not None:
                        self._coef = best_coef
                        self._intercept = best_intercept
                    break
            else:
                if len(self.loss) > 1 and abs(self.loss[-2] - self.loss[-1]) < self.tol:
                    break

    def predict_proba(self, x):
        x = np.asarray(x)
        scores = x.dot(self.coef_.T) + self.intercept_
        return self.softmax(scores)

    def predict(self, x):
        proba = self.predict_proba(x)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def coef_(self):
        return self._coef

    @property
    def intercept_(self):
        return self._intercept

    @coef_.setter
    def coef_(self, value):
        self._coef = value

    @intercept_.setter
    def intercept_(self, value):
        self._intercept = value
