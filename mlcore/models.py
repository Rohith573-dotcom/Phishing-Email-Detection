"""
models.py
=========
Classifiers implemented FROM SCRATCH with NumPy only - no scikit-learn.

    KNNClassifier               non-parametric, cosine distance
    LogisticRegression          parametric, full-batch gradient descent + L2
    DecisionTreeClassifier      CART with Gini impurity
    RandomForestClassifier      bagging + random feature subspace
    NeuralNetwork               1 hidden layer MLP, mini-batch Adam
    MultinomialNB               generative baseline for text (Project 2)

Every estimator exposes fit / predict / predict_proba so they are drop-in
compatible with the evaluation harness and with scikit-learn's API.

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import numpy as np


# =========================================================================== #
# 1. K-Nearest Neighbours  (NON-PARAMETRIC)
# =========================================================================== #
class KNNClassifier:
    """Lazy learner: no parameters are estimated at fit time.

    Distance
    --------
    Cosine distance (1 - cosine similarity) is used instead of Euclidean.
    On L2-normalised TF-IDF vectors, cosine measures the *angle* between
    documents and is therefore insensitive to document length - a long and a
    short article about the same topic stay close together, which plain
    Euclidean distance would not guarantee.

    Complexity
    ----------
    Training  O(1).   Prediction  O(n_train x n_features) per query - the
    reason KNN is the slowest model at inference time in the results table.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y).astype(int)
        norms = np.linalg.norm(self.X_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.Xn_ = self.X_ / norms
        self.classes_ = np.unique(self.y_)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sim = (X / norms) @ self.Xn_.T                      # cosine similarity
        k = min(self.n_neighbors, self.Xn_.shape[0])
        idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]

        proba = np.zeros((X.shape[0], len(self.classes_)))
        rows = np.arange(X.shape[0])[:, None]
        neigh_sim = sim[rows, idx]
        neigh_lab = self.y_[idx]
        w = np.clip(neigh_sim, 0.0, None) + 1e-9 if self.weights == "distance" \
            else np.ones_like(neigh_sim)
        for c, cls in enumerate(self.classes_):
            proba[:, c] = (w * (neigh_lab == cls)).sum(axis=1)
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# =========================================================================== #
# 2. Logistic Regression  (PARAMETRIC)
# =========================================================================== #
class LogisticRegression:
    """Binary logistic regression trained by gradient descent.

    Model      p(y=1|x) = sigma(w.x + b),  sigma(z) = 1 / (1 + e^-z)
    Loss       mean binary cross-entropy + (lambda/2)||w||^2
    Gradient   dL/dw = X^T (p - y) / n + lambda*w
    """

    def __init__(self, lr: float = 0.5, n_iters: int = 600,
                 l2: float = 1e-4, verbose: bool = False):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.verbose = verbose

    @staticmethod
    def _sigmoid(z):
        # Numerically stable: avoids exp overflow for large |z|.
        out = np.empty_like(z)
        pos, neg = z >= 0, z < 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1.0 + ez)
        return out

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history_ = []

        for it in range(self.n_iters):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            grad_w = X.T @ err / n + self.l2 * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            eps = 1e-12
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history_.append(loss)
            if self.verbose and it % 100 == 0:
                print(f"    iter {it:4d}   loss {loss:.4f}")

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = self._sigmoid(np.asarray(X, dtype=np.float64) @ self.w + self.b)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def top_features(self, names: list[str], k: int = 15):
        """Most positive / most negative coefficients - drives interpretability."""
        order = np.argsort(self.w)
        neg = [(names[i], float(self.w[i])) for i in order[:k]]
        pos = [(names[i], float(self.w[i])) for i in order[::-1][:k]]
        return pos, neg


# =========================================================================== #
# 3. Decision Tree (CART, Gini)
# =========================================================================== #
class DecisionTreeClassifier:
    """Binary CART tree split on Gini impurity.

        Gini(S) = 1 - sum_c p_c^2
        Gain    = Gini(parent) - weighted mean Gini(children)

    Candidate thresholds are drawn from feature quantiles rather than every
    unique value: on sparse TF-IDF data this cuts fit time by roughly an order
    of magnitude with no measurable loss of accuracy.
    """

    def __init__(self, max_depth: int = 12, min_samples_split: int = 4,
                 max_features: str | int | None = "sqrt",
                 n_thresholds: int = 8, random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_thresholds = n_thresholds
        self.random_state = random_state

    # -- impurity --------------------------------------------------------- #
    @staticmethod
    def _gini(y) -> float:
        if y.size == 0:
            return 0.0
        p = np.bincount(y, minlength=2) / y.size
        return 1.0 - np.sum(p ** 2)

    def _n_feats(self, d: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(d)))
        if self.max_features == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(self.max_features, int):
            return min(self.max_features, d)
        return d

    # -- recursive construction ------------------------------------------- #
    def _build(self, X, y, depth):
        node = {"leaf": True, "proba": np.bincount(y, minlength=2) / max(len(y), 1)}

        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return node

        n, d = X.shape
        feats = self.rng_.choice(d, size=self._n_feats(d), replace=False)
        parent = self._gini(y)
        best = (0.0, None, None)                     # gain, feature, threshold

        for f in feats:
            col = X[:, f]
            qs = np.quantile(col, np.linspace(0.1, 0.9, self.n_thresholds))
            for thr in np.unique(qs):
                mask = col <= thr
                nl = int(mask.sum())
                if nl == 0 or nl == n:
                    continue
                gain = parent - (nl / n) * self._gini(y[mask]) \
                              - ((n - nl) / n) * self._gini(y[~mask])
                if gain > best[0]:
                    best = (gain, f, float(thr))

        if best[1] is None or best[0] <= 1e-9:
            return node

        _, f, thr = best
        mask = X[:, f] <= thr
        return {"leaf": False, "feature": f, "threshold": thr,
                "left": self._build(X[mask], y[mask], depth + 1),
                "right": self._build(X[~mask], y[~mask], depth + 1)}

    def fit(self, X, y):
        self.rng_ = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        self.n_features_ = X.shape[1]
        self.tree_ = self._build(X, y, 0)
        self.classes_ = np.array([0, 1])
        return self

    def _walk(self, node, x):
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["proba"]

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return np.vstack([self._walk(self.tree_, x) for x in X])

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_counts(self) -> np.ndarray:
        """How often each feature was chosen as a split - basis of importance."""
        counts = np.zeros(self.n_features_)

        def rec(node):
            if node["leaf"]:
                return
            counts[node["feature"]] += 1
            rec(node["left"])
            rec(node["right"])

        rec(self.tree_)
        return counts


# =========================================================================== #
# 4. Random Forest (ENSEMBLE)
# =========================================================================== #
class RandomForestClassifier:
    """Bagged decision trees with a random feature subspace at every split.

    Two independent sources of decorrelation:
      1. each tree sees a bootstrap resample of the rows;
      2. each split considers only sqrt(d) randomly chosen columns.

    Averaging decorrelated high-variance trees is what drives the variance
    reduction that makes the ensemble beat any single tree.
    """

    def __init__(self, n_estimators: int = 40, max_depth: int = 14,
                 max_features: str | int | None = "sqrt",
                 min_samples_split: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        self.trees_ = []

        for b in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)              # bootstrap sample
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 1_000_000)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)

        self.n_features_ = X.shape[1]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([t.predict_proba(X) for t in self.trees_], axis=0)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_importances_(self) -> np.ndarray:
        imp = np.sum([t.feature_counts() for t in self.trees_], axis=0)
        return imp / max(imp.sum(), 1.0)


# =========================================================================== #
# 5. Neural Network (DEEP LEARNING)
# =========================================================================== #
class NeuralNetwork:
    """One hidden layer MLP:  input -> ReLU(h) -> sigmoid(1).

    Trained with mini-batch Adam and inverted dropout.  Forward and backward
    passes are written out explicitly so the chain rule is visible rather
    than hidden behind a framework.
    """

    def __init__(self, hidden: int = 128, lr: float = 3e-3, epochs: int = 40,
                 batch_size: int = 64, l2: float = 1e-5, dropout: float = 0.2,
                 random_state: int = 42, verbose: bool = False):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.dropout = dropout
        self.random_state = random_state
        self.verbose = verbose

    def _init_params(self, d):
        rng = np.random.default_rng(self.random_state)
        # He initialisation keeps the variance of activations stable through ReLU.
        self.W1 = rng.normal(0, np.sqrt(2.0 / d), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden), (self.hidden, 1))
        self.b2 = np.zeros(1)
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._t = 0
        self.rng_ = rng

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def _adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = beta1 * self._m[k] + (1 - beta1) * g
            self._v[k] = beta2 * self._v[k] + (1 - beta2) * (g ** 2)
            mhat = self._m[k] / (1 - beta1 ** self._t)
            vhat = self._v[k] / (1 - beta2 ** self._t)
            setattr(self, k, getattr(self, k) - self.lr * mhat / (np.sqrt(vhat) + eps))

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        n, d = X.shape
        self._init_params(d)
        self.loss_history_ = []

        for ep in range(self.epochs):
            order = self.rng_.permutation(n)
            epoch_loss = 0.0
            for s in range(0, n, self.batch_size):
                bi = order[s:s + self.batch_size]
                xb, yb = X[bi], y[bi]
                m = len(bi)

                # ---- forward ------------------------------------------- #
                z1 = xb @ self.W1 + self.b1
                a1 = np.maximum(0.0, z1)                       # ReLU
                if self.dropout > 0:
                    mask = (self.rng_.random(a1.shape) > self.dropout) / (1 - self.dropout)
                    a1 = a1 * mask
                else:
                    mask = None
                z2 = a1 @ self.W2 + self.b2
                p = self._sigmoid(z2)

                eps = 1e-12
                epoch_loss += -np.mean(yb * np.log(p + eps) +
                                       (1 - yb) * np.log(1 - p + eps)) * m

                # ---- backward ------------------------------------------ #
                dz2 = (p - yb) / m                              # dL/dz2
                gW2 = a1.T @ dz2 + self.l2 * self.W2
                gb2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                if mask is not None:
                    da1 = da1 * mask
                dz1 = da1 * (z1 > 0)                            # ReLU derivative
                gW1 = xb.T @ dz1 + self.l2 * self.W1
                gb1 = dz1.sum(axis=0)

                self._adam({"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2})

            self.loss_history_.append(epoch_loss / n)
            if self.verbose and ep % 10 == 0:
                print(f"    epoch {ep:3d}   loss {self.loss_history_[-1]:.4f}")

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        a1 = np.maximum(0.0, X @ self.W1 + self.b1)             # no dropout at test
        p = self._sigmoid(a1 @ self.W2 + self.b2).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# =========================================================================== #
# 6. Multinomial Naive Bayes  (generative baseline - Project 2)
# =========================================================================== #
class MultinomialNB:
    """Naive Bayes for count / TF-IDF features with Laplace smoothing.

        log p(y|x)  ~  log p(y) + sum_t x_t * log p(t|y)

    "Naive" because it assumes every term is conditionally independent given
    the class.  That assumption is plainly false for language, yet the model
    remains a strong, near-instant baseline for spam and phishing filters.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        counts = np.vstack([X[y == c].sum(axis=0) for c in self.classes_]) + self.alpha
        self.feature_log_prob_ = np.log(counts / counts.sum(axis=1, keepdims=True))
        self.class_log_prior_ = np.log(
            np.array([(y == c).mean() for c in self.classes_]))
        return self

    def predict_proba(self, X) -> np.ndarray:
        jll = np.asarray(X, dtype=np.float64) @ self.feature_log_prob_.T \
              + self.class_log_prior_
        jll -= jll.max(axis=1, keepdims=True)
        e = np.exp(jll)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
