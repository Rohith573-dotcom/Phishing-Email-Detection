"""
features.py
===========
Structural / metadata feature extraction for phishing e-mail detection.

Text alone is not the whole signal in a phishing message.  A lure is also
recognisable by its *shape*: how many links it carries, whether those links
point at a raw IP address, whether the sender is a role account on a
hyphenated look-alike domain, whether the salutation is generic, how much
of the subject line is shouted in capitals.

This module turns each raw e-mail into a fixed vector of 17 interpretable
numeric features.  Everything is computed with the standard library and
NumPy - no external NLP package.

    extract_features(sender, subject, body) -> dict
    build_matrix(df)                        -> (X, feature_names)
    StandardScaler                          -> zero-mean / unit-variance

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Lexicons (defensive detection cues - deliberately small and auditable)
# --------------------------------------------------------------------------- #
URGENCY_WORDS = {
    "urgent", "immediately", "immediate", "now", "today", "expires", "expiring",
    "final", "asap", "act", "hurry", "deadline", "suspend", "suspended",
    "terminate", "revoked", "restricted", "limited", "warning", "alert",
}
CREDENTIAL_WORDS = {
    "password", "passcode", "otp", "login", "log", "signin", "credential",
    "credentials", "verify", "verification", "confirm", "authenticate",
    "account", "ssn", "pin", "billing", "card", "cvv", "bank",
}
REWARD_WORDS = {
    "winner", "won", "prize", "reward", "bonus", "gift", "claim",
    "congratulations", "selected", "free", "refund",
}
ROLE_ACCOUNTS = {
    "support", "no-reply", "noreply", "security", "admin", "admin-team",
    "service-desk", "billing", "verification", "alerts", "helpdesk",
    "postmaster", "notification", "notifications",
}
GENERIC_GREETINGS = (
    "dear customer", "dear user", "dear sir", "dear madam", "dear account holder",
    "dear client", "dear member", "valued customer", "dear subscriber",
)

_RE_URL    = re.compile(r"https?://[^\s\"'<>]+", re.I)
_RE_IPHOST = re.compile(r"https?://(?:\d{1,3}\.){3}\d{1,3}", re.I)
_RE_ANCHOR = re.compile(r"<a\s[^>]*href=", re.I)
_RE_WORD   = re.compile(r"[A-Za-z']+")

FEATURE_NAMES = [
    "n_urls", "has_ip_url", "n_insecure_urls", "max_url_path_depth",
    "has_html_anchor", "sender_domain_hyphens", "sender_subdomain_depth",
    "sender_is_role_account", "sender_domain_len", "n_exclamations",
    "subject_caps_ratio", "subject_urgency_hits", "body_urgency_hits",
    "credential_hits", "reward_hits", "generic_greeting", "body_word_count",
]


# --------------------------------------------------------------------------- #
def extract_features(sender: str, subject: str, body: str) -> dict[str, float]:
    """Return the 17-dimensional metadata vector for one e-mail."""
    sender  = sender  if isinstance(sender, str)  else ""
    subject = subject if isinstance(subject, str) else ""
    body    = body    if isinstance(body, str)    else ""

    # ---- link structure ------------------------------------------------- #
    urls = _RE_URL.findall(body)
    depths = [u.split("://", 1)[-1].count("/") for u in urls] or [0]

    # ---- sender anatomy ------------------------------------------------- #
    user, _, domain = sender.partition("@")
    labels = [l for l in domain.split(".") if l]

    # ---- lexical counts ------------------------------------------------- #
    body_words    = [w.lower() for w in _RE_WORD.findall(body)]
    subject_words = [w.lower() for w in _RE_WORD.findall(subject)]
    subj_letters  = [c for c in subject if c.isalpha()]

    return {
        "n_urls":                 float(len(urls)),
        "has_ip_url":             float(bool(_RE_IPHOST.search(body))),
        "n_insecure_urls":        float(sum(1 for u in urls if u.lower().startswith("http://"))),
        "max_url_path_depth":     float(max(depths)),
        "has_html_anchor":        float(bool(_RE_ANCHOR.search(body))),
        "sender_domain_hyphens":  float(domain.count("-")),
        "sender_subdomain_depth": float(max(len(labels) - 2, 0)),
        "sender_is_role_account": float(user.lower() in ROLE_ACCOUNTS),
        "sender_domain_len":      float(len(domain)),
        "n_exclamations":         float(body.count("!") + subject.count("!")),
        "subject_caps_ratio":     float(sum(c.isupper() for c in subj_letters) /
                                        max(len(subj_letters), 1)),
        "subject_urgency_hits":   float(sum(w in URGENCY_WORDS for w in subject_words)),
        "body_urgency_hits":      float(sum(w in URGENCY_WORDS for w in body_words)),
        "credential_hits":        float(sum(w in CREDENTIAL_WORDS for w in body_words)),
        "reward_hits":            float(sum(w in REWARD_WORDS for w in body_words)),
        "generic_greeting":       float(any(g in body.lower() for g in GENERIC_GREETINGS)),
        "body_word_count":        float(len(body_words)),
    }


def build_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Vectorise a whole DataFrame of e-mails into the metadata matrix."""
    rows = [extract_features(s, su, b)
            for s, su, b in zip(df["sender"], df["subject"], df["body"])]
    X = np.array([[r[k] for k in FEATURE_NAMES] for r in rows], dtype=np.float32)
    return X, list(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
class StandardScaler:
    """z-score normalisation, fitted on the training split only.

    Metadata features live on wildly different scales (a binary flag versus a
    300-word body count).  Distance-based and gradient-based learners need
    them on a common scale or the large-magnitude columns dominate.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mean_) /
                self.scale_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Rescale every column into [0, 1] using training-split extremes.

    Used in preference to z-scoring for the combined feature matrix because
    Multinomial Naive Bayes is only defined for non-negative inputs, and the
    L2-normalised TF-IDF block it is concatenated with is non-negative too -
    so both blocks end up on the same footing.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.min_ = X.min(axis=0)
        self.range_ = X.max(axis=0) - self.min_
        self.range_[self.range_ == 0] = 1.0
        return self

    def transform(self, X):
        Z = (np.asarray(X, dtype=np.float64) - self.min_) / self.range_
        return np.clip(Z, 0.0, 1.0).astype(np.float32)   # clip unseen test extremes

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
def clean_email_body(body: str) -> str:
    """Week-2 cleaning for the *text* channel.

    HTML tags and URLs are stripped here because their structure has already
    been captured numerically above - leaving raw URLs in the text channel
    would flood the vocabulary with one-off tokens.
    """
    body = body if isinstance(body, str) else ""
    body = re.sub(r"<[^>]+>", " ", body)
    body = _RE_URL.sub(" ", body)
    body = re.sub(r"\s+", " ", body)
    return body.strip()
