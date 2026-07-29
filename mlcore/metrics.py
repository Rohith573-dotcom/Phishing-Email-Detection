"""
metrics.py
==========
Evaluation metrics implemented FROM SCRATCH, plus matplotlib helpers used to
produce the figures in the report.

    confusion_matrix, accuracy, precision, recall, f1, specificity
    roc_curve, roc_auc, classification_report
    k_fold_indices                 - manual stratified k-fold
    plot_confusion_matrix, plot_model_comparison, plot_roc_curves,
    plot_learning_curve, plot_top_features

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib

# Force the headless backend only when this module is imported by a plain
# script.  Inside a Jupyter kernel the inline backend is already active and
# overriding it here would silently suppress every figure in the notebook.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1F3A5F", "#E4572E", "#17A398", "#F3A712", "#7D5BA6"]


# --------------------------------------------------------------------------- #
# Core counting metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Return [[TN, FP], [FN, TP]] for binary labels in {0, 1}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision(y_true, y_pred) -> float:
    """TP / (TP + FP) - of everything flagged positive, how much really was."""
    _, fp, _, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fp)) if (tp + fp) else 0.0


def recall(y_true, y_pred) -> float:
    """TP / (TP + FN) - of all true positives, how many were caught."""
    _, _, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp / (tp + fn)) if (tp + fn) else 0.0


def specificity(y_true, y_pred) -> float:
    """TN / (TN + FP) - the true-negative rate."""
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else 0.0


def f1(y_true, y_pred) -> float:
    """Harmonic mean of precision and recall."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return float(2 * p * r / (p + r)) if (p + r) else 0.0


def roc_curve(y_true, scores):
    """Sweep the decision threshold and return (fpr, tpr) arrays."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    y = y_true[order]
    P, N = y.sum(), len(y) - y.sum()
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tps / max(P, 1), [1.0]])
    fpr = np.concatenate([[0.0], fps / max(N, 1), [1.0]])
    return fpr, tpr


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve via the trapezoid rule."""
    fpr, tpr = roc_curve(y_true, scores)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") \
        else float(np.trapz(tpr, fpr))


def classification_report(y_true, y_pred, labels=("Class 0", "Class 1")) -> str:
    """Per-class precision / recall / F1 table, printed like sklearn's."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    lines = [f"{'':<14}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>10}", ""]
    for c, name in enumerate(labels):
        t = (y_true == c).astype(int)
        p = (y_pred == c).astype(int)
        pr, rc = precision(t, p), recall(t, p)
        f = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        lines.append(f"{name:<14}{pr:>10.3f}{rc:>10.3f}{f:>10.3f}{int(t.sum()):>10d}")
    lines += ["", f"{'accuracy':<14}{'':>10}{'':>10}{accuracy(y_true, y_pred):>10.3f}"
                  f"{len(y_true):>10d}"]
    return "\n".join(lines)


def k_fold_indices(y, k: int = 5, random_state: int = 42):
    """Yield (train_idx, val_idx) for stratified k-fold cross-validation."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        for i, v in enumerate(idx):
            folds[i % k].append(v)
    folds = [np.array(sorted(f)) for f in folds]
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield np.sort(train), val


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, labels, title, path, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=11, weight="bold")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                    fontsize=15, weight="bold",
                    color="white" if cm[i, j] > thresh else "#1F3A5F")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_model_comparison(results: dict, path: str, title: str):
    """Grouped bar chart of accuracy / precision / recall / F1 per model."""
    models = list(results)
    metrics = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(models))
    w = 0.2

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, m in enumerate(metrics):
        vals = [results[k][m] for k in models]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m.capitalize(),
                      color=PALETTE[i], edgecolor="white")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}",
                    ha="center", fontsize=6.5)
    ax.set_xticks(x, models, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, weight="bold")
    ax.legend(ncol=4, frameon=False, fontsize=9, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_roc_curves(curves: dict, path: str, title: str):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)], lw=2,
                label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_learning_curve(histories: dict, path: str, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for i, (name, h) in enumerate(histories.items()):
        ax.plot(h, color=PALETTE[i % len(PALETTE)], lw=1.8, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training loss")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_top_features(pos, neg, path, title, pos_label, neg_label):
    """Horizontal diverging bar chart of the most influential terms."""
    names = [n for n, _ in neg][::-1] + [n for n, _ in pos][::-1]
    vals = [v for _, v in neg][::-1] + [v for _, v in pos][::-1]
    colors = ["#17A398"] * len(neg) + ["#E4572E"] * len(pos)

    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("Logistic-regression coefficient")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#E4572E"),
               plt.Rectangle((0, 0), 1, 1, color="#17A398")]
    ax.legend(handles, [pos_label, neg_label], fontsize=8, frameon=False,
              loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_class_balance(counts: dict, path: str, title: str):
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    bars = ax.bar(list(counts), list(counts.values()),
                  color=[PALETTE[0], PALETTE[1]], width=0.55)
    for b, v in zip(bars, counts.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + max(counts.values()) * 0.02,
                str(v), ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("Number of documents")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
