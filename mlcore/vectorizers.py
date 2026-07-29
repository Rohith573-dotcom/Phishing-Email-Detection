"""
vectorizers.py
==============
Feature-extraction implemented FROM SCRATCH with NumPy only.

    CountVectorizer   - Bag-of-Words term counts
    TfidfVectorizer   - TF x IDF with smoothing + L2 normalisation
    LSAEmbedding      - dense document embeddings via truncated SVD of the
                        TF-IDF matrix (Latent Semantic Analysis)

All three expose the familiar fit / transform / fit_transform API so they can
be swapped for the scikit-learn equivalents during validation.

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

from collections import Counter
import numpy as np


# --------------------------------------------------------------------------- #
class CountVectorizer:
    """Bag-of-Words.

    Each document becomes a vector of raw term counts over a vocabulary that
    is learned from the training corpus only (never from the test set - that
    would be data leakage).
    """

    def __init__(self, max_features: int = 5000, min_df: int = 2,
                 ngram_range: tuple[int, int] = (1, 1)):
        self.max_features = max_features
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------- #
    def _ngrams(self, tokens: list[str]) -> list[str]:
        lo, hi = self.ngram_range
        out: list[str] = []
        for n in range(lo, hi + 1):
            if n == 1:
                out.extend(tokens)
            else:
                out.extend("_".join(tokens[i:i + n])
                           for i in range(len(tokens) - n + 1))
        return out

    # -- API -------------------------------------------------------------- #
    def fit(self, docs: list[list[str]]):
        """Learn the vocabulary from tokenised documents."""
        df = Counter()          # document frequency
        tf = Counter()          # corpus term frequency
        for tokens in docs:
            grams = self._ngrams(tokens)
            tf.update(grams)
            df.update(set(grams))

        candidates = [(t, tf[t]) for t, d in df.items() if d >= self.min_df]
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        candidates = candidates[: self.max_features]

        self.vocabulary_ = {t: i for i, t in enumerate(sorted(c[0] for c in candidates))}
        self.document_frequency_ = np.array(
            [df[t] for t, _ in sorted(self.vocabulary_.items(), key=lambda kv: kv[1])],
            dtype=np.float64)
        self.n_docs_ = len(docs)
        return self

    def transform(self, docs: list[list[str]]) -> np.ndarray:
        X = np.zeros((len(docs), len(self.vocabulary_)), dtype=np.float32)
        for r, tokens in enumerate(docs):
            for g, c in Counter(self._ngrams(tokens)).items():
                j = self.vocabulary_.get(g)
                if j is not None:
                    X[r, j] = c
        return X

    def fit_transform(self, docs: list[list[str]]) -> np.ndarray:
        return self.fit(docs).transform(docs)

    def get_feature_names(self) -> list[str]:
        return [t for t, _ in sorted(self.vocabulary_.items(), key=lambda kv: kv[1])]


# --------------------------------------------------------------------------- #
class TfidfVectorizer(CountVectorizer):
    """Term Frequency - Inverse Document Frequency.

        tf(t,d)   = count(t,d) / |d|                 (sub-linear option below)
        idf(t)    = ln((1 + N) / (1 + df(t))) + 1    (smoothed, sklearn-style)
        tfidf     = tf * idf, then L2-normalised per document

    IDF down-weights words that appear everywhere ("said", "report") and
    boosts words that are concentrated in a few documents - exactly the
    behaviour needed to separate sensational fake-news vocabulary from the
    neutral vocabulary of wire copy.
    """

    def __init__(self, max_features: int = 5000, min_df: int = 2,
                 ngram_range: tuple[int, int] = (1, 1), sublinear_tf: bool = True):
        super().__init__(max_features, min_df, ngram_range)
        self.sublinear_tf = sublinear_tf

    def fit(self, docs: list[list[str]]):
        super().fit(docs)
        self.idf_ = np.log((1.0 + self.n_docs_) /
                           (1.0 + self.document_frequency_)) + 1.0
        return self

    def transform(self, docs: list[list[str]]) -> np.ndarray:
        counts = super().transform(docs)

        if self.sublinear_tf:
            tf = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1e-9)), 0.0)
        else:
            lengths = counts.sum(axis=1, keepdims=True)
            lengths[lengths == 0] = 1.0
            tf = counts / lengths

        X = tf * self.idf_
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (X / norms).astype(np.float32)

    def fit_transform(self, docs: list[list[str]]) -> np.ndarray:
        return self.fit(docs).transform(docs)

    def top_terms(self, k: int = 20) -> list[tuple[str, float]]:
        """Highest-IDF (most discriminative) terms - useful for the EDA section."""
        names = self.get_feature_names()
        order = np.argsort(-self.idf_)[:k]
        return [(names[i], float(self.idf_[i])) for i in order]


# --------------------------------------------------------------------------- #
class LSAEmbedding:
    """Dense document embeddings via truncated SVD (Latent Semantic Analysis).

    The TF-IDF matrix is factorised as  X ~ U S V^T.  Keeping the leading
    `n_components` singular directions gives every document a short dense
    vector in which synonyms collapse onto nearby axes - a classic and fully
    self-contained alternative to downloading pre-trained Word2Vec/GloVe
    vectors (which the brief rules out as a "pre-built solution").
    """

    def __init__(self, n_components: int = 100, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray):
        Xc = np.asarray(X, dtype=np.float64)
        self.mean_ = Xc.mean(axis=0)
        # Randomised range finder keeps SVD tractable on wide TF-IDF matrices.
        rng = np.random.default_rng(self.random_state)
        k = min(self.n_components, min(Xc.shape) - 1)
        Omega = rng.standard_normal((Xc.shape[1], k + 10))
        Y = Xc @ Omega
        Q, _ = np.linalg.qr(Y)
        B = Q.T @ Xc
        Ub, S, Vt = np.linalg.svd(B, full_matrices=False)
        self.components_ = Vt[:k]
        self.singular_values_ = S[:k]
        self.explained_variance_ratio_ = (S[:k] ** 2) / np.sum(S ** 2)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
