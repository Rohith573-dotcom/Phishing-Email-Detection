"""
preprocessing.py
================
Text preprocessing utilities implemented FROM SCRATCH (no NLTK / no spaCy).

Provided:
    STOPWORDS          - curated English stop-word set
    clean_text         - lowercase, strip HTML/URLs/punctuation/digits
    tokenize           - whitespace + regex word tokenizer
    remove_stopwords   - stop-word filter
    porter_lite_stem   - lightweight suffix-stripping stemmer
    preprocess         - full pipeline (clean -> tokenize -> stop -> stem)
    train_test_split   - stratified shuffle split

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import re
import numpy as np

# --------------------------------------------------------------------------- #
# 1. Stop-word list (built manually - no external corpus download required)
# --------------------------------------------------------------------------- #
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn",
    "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn", "has", "hasn",
    "have", "haven", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "it", "its",
    "itself", "just", "ll", "me", "more", "most", "mustn", "my", "myself", "no",
    "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same",
    "shan", "she", "should", "shouldn", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "ve", "very", "was", "wasn", "we", "were", "weren", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "would", "wouldn", "you", "your", "yours", "yourself", "yourselves",
}

# --------------------------------------------------------------------------- #
# 2. Regex patterns compiled once (module import time) for speed
# --------------------------------------------------------------------------- #
_RE_HTML   = re.compile(r"<[^>]+>")
_RE_URL    = re.compile(r"http\S+|www\.\S+")
_RE_EMAIL  = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_RE_NONAL  = re.compile(r"[^a-z\s]")
_RE_SPACE  = re.compile(r"\s+")
_RE_TOKEN  = re.compile(r"[a-z]{2,}")


def clean_text(text: str,
               strip_urls: bool = True,
               strip_html: bool = True) -> str:
    """Lowercase and remove HTML tags, URLs, e-mails, digits and punctuation.

    Parameters
    ----------
    text : str
        Raw document.
    strip_urls, strip_html : bool
        Toggle individual cleaning stages (kept configurable so the phishing
        project can *keep* URL information as a separate metadata feature).

    Returns
    -------
    str : normalised text with single spaces.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    if strip_html:
        text = _RE_HTML.sub(" ", text)
    if strip_urls:
        text = _RE_URL.sub(" ", text)
        text = _RE_EMAIL.sub(" ", text)
    text = _RE_NONAL.sub(" ", text)          # drops punctuation + digits
    text = _RE_SPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens of length >= 2."""
    return _RE_TOKEN.findall(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Drop high-frequency function words that carry little signal."""
    return [t for t in tokens if t not in STOPWORDS]


def porter_lite_stem(word: str) -> str:
    """A compact suffix-stripping stemmer.

    This is a simplified re-implementation of the first steps of the Porter
    algorithm.  It is deliberately conservative: short words are never
    stemmed, which avoids the over-stemming that hurts precision on
    short news headlines.
    """
    if len(word) <= 3:
        return word
    for suf, repl in (("ational", "ate"), ("tional", "tion"), ("iveness", "ive"),
                      ("fulness", "ful"), ("ousness", "ous"), ("ization", "ize"),
                      ("ation", "ate"), ("ements", "ement"), ("ement", "ement"),
                      ("ingly", ""), ("edly", ""), ("ies", "y"), ("sses", "ss"),
                      ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""), ("s", "")):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: len(word) - len(suf)] + repl
    return word


def preprocess(text: str,
               stem: bool = True,
               drop_stopwords: bool = True,
               strip_urls: bool = True) -> list[str]:
    """Run the full Week-1 pipeline and return a token list."""
    tokens = tokenize(clean_text(text, strip_urls=strip_urls))
    if drop_stopwords:
        tokens = remove_stopwords(tokens)
    if stem:
        tokens = [porter_lite_stem(t) for t in tokens]
    return tokens


# --------------------------------------------------------------------------- #
# 3. Stratified train / test split (implemented manually)
# --------------------------------------------------------------------------- #
def train_test_split(X, y, test_size: float = 0.2, random_state: int = 42,
                     stratify: bool = True):
    """Shuffle-split arrays into train / test partitions.

    A stratified split keeps the class ratio of the full corpus inside both
    partitions, which matters because a naive random split can leave the
    minority class badly under-represented in the test set.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(y)

    if not stratify:
        idx = rng.permutation(n)
        cut = int(n * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
    else:
        tr, te = [], []
        for cls in np.unique(y):
            cls_idx = np.flatnonzero(y == cls)
            rng.shuffle(cls_idx)
            cut = int(len(cls_idx) * (1 - test_size))
            tr.extend(cls_idx[:cut])
            te.extend(cls_idx[cut:])
        tr, te = np.array(tr), np.array(te)
        rng.shuffle(tr)
        rng.shuffle(te)

    return X[tr], X[te], y[tr], y[te]
