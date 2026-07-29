"""
main.py  -  AI-Driven Phishing Email Detection Using NLP
========================================================
End-to-end pipeline for Project 2 of the IICT Summer Internship 2026.

The system combines two complementary signals:

  * a LINGUISTIC channel - manual tokenisation, stop-word removal, stemming
    and TF-IDF over unigrams and bigrams of the subject + body;
  * a STRUCTURAL channel - 17 interpretable metadata features describing the
    links, the sender anatomy and the pressure tactics of each message.

Four classifiers are trained from scratch (Logistic Regression, Random Forest,
Multinomial Naive Bayes and a one-hidden-layer Neural Network) and each is
evaluated on three feature views, which answers the central question of the
comparative report: how much does metadata add on top of text?

Run
---
    python generate_sample_dataset.py     # once, if data/ is empty
    python main.py

Outputs
-------
    data/emails_clean.csv   cleaned corpus (deliverable: raw + cleaned dataset)
    results/*.png           figures for the report and the slide deck
    results/metrics.csv     full results table
    results/ablation.csv    text vs metadata vs combined
    results/metrics.json    machine-readable results
    results/summary.txt     console log

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlcore.preprocessing import preprocess, train_test_split
from mlcore.vectorizers import TfidfVectorizer
from mlcore.models import (LogisticRegression, RandomForestClassifier,
                           NeuralNetwork, MultinomialNB)
from mlcore import metrics as M
import features as F

# --------------------------------------------------------------------------- #
CONFIG = {
    "max_features": 1200,
    "min_df":       3,
    "ngram_range":  (1, 2),
    "test_size":    0.20,
    "random_state": 42,
    "rf_trees":     30,
    "rf_depth":     10,
    "nn_hidden":    96,
    "nn_epochs":    40,
    "cv_folds":     5,
}
LABELS = ["LEGITIMATE (0)", "PHISHING (1)"]
RESULTS = "results"


def log(msg: str, buf: list[str]):
    print(msg)
    buf.append(msg)


# --------------------------------------------------------------------------- #
# PHASE 1  -  Data collection
# --------------------------------------------------------------------------- #
def load_dataset(buf) -> pd.DataFrame:
    for path, note in (("data/emails.csv", "user-supplied corpus"),
                       ("data/emails_raw.csv", "synthetic sample corpus")):
        if os.path.exists(path):
            df = pd.read_csv(path)
            log(f"[Phase 1] Loaded {path}  ({note})", buf)
            break
    else:
        raise FileNotFoundError("No dataset found. Run generate_sample_dataset.py first.")

    for col in ("sender", "subject", "body", "label"):
        if col not in df.columns:
            raise ValueError(f"Dataset must contain a '{col}' column.")

    before = len(df)
    df = df.dropna(subset=["body", "label"]).drop_duplicates(subset=["body"])
    df["label"] = df["label"].astype(int)
    log(f"[Phase 1] Dropped {before - len(df)} null/duplicate rows -> {len(df)} e-mails", buf)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# PHASE 2  -  Data cleaning
# --------------------------------------------------------------------------- #
def clean_corpus(df: pd.DataFrame, buf) -> pd.DataFrame:
    """Strip HTML and URLs from the text channel and persist the clean corpus."""
    t0 = time.time()
    df = df.copy()
    df["clean_subject"] = df["subject"].astype(str).map(F.clean_email_body)
    df["clean_body"] = df["body"].astype(str).map(F.clean_email_body)
    df["clean_text"] = df["clean_subject"] + " " + df["clean_body"]
    # Persist only the cleaned columns: the raw subject/body already live in the
    # raw corpus, and duplicating them here tripled the size of the file.
    df[["id", "sender", "clean_subject", "clean_body", "clean_text", "label"]] \
        .to_csv("data/emails_clean.csv", index=False)
    log(f"[Phase 2] Cleaned corpus written to data/emails_clean.csv "
        f"({time.time() - t0:.1f}s)", buf)

    raw_len = df["body"].astype(str).str.len().mean()
    cl_len = df["clean_body"].str.len().mean()
    log(f"[Phase 2] Mean body length: {raw_len:.0f} chars raw -> {cl_len:.0f} chars cleaned "
        f"({100 * (1 - cl_len / raw_len):.1f}% removed as markup/URLs)", buf)
    return df


# --------------------------------------------------------------------------- #
# PHASE 3  -  Feature engineering
# --------------------------------------------------------------------------- #
def metadata_report(df: pd.DataFrame, Xm: np.ndarray, names: list[str], buf):
    """Class-conditional means for every structural feature."""
    y = df["label"].to_numpy()
    rows = []
    for j, n in enumerate(names):
        legit, phish = Xm[y == 0, j].mean(), Xm[y == 1, j].mean()
        rows.append({"feature": n, "legitimate_mean": round(float(legit), 3),
                     "phishing_mean": round(float(phish), 3),
                     "ratio": round(float((phish + 1e-6) / (legit + 1e-6)), 2)})
    tab = pd.DataFrame(rows).sort_values("ratio", ascending=False)
    tab.to_csv(f"{RESULTS}/metadata_profile.csv", index=False)
    log("\n[Phase 3] Structural feature profile (class-conditional means)", buf)
    log(tab.to_string(index=False), buf)

    top = tab.head(8).iloc[::-1]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    yy = np.arange(len(top))
    ax.barh(yy - 0.19, top["legitimate_mean"], 0.38, label="Legitimate", color="#1F3A5F")
    ax.barh(yy + 0.19, top["phishing_mean"], 0.38, label="Phishing", color="#E4572E")
    ax.set_yticks(yy, top["feature"], fontsize=9)
    ax.set_xlabel("Mean value per e-mail")
    ax.set_title("Structural features that separate the classes",
                 fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig2_metadata_profile.png", dpi=170)
    plt.close(fig)
    return tab


# --------------------------------------------------------------------------- #
# PHASE 4 / 5  -  Model development and evaluation
# --------------------------------------------------------------------------- #
def make_models():
    return {
        "Logistic Regression": LogisticRegression(lr=1.5, n_iters=1500),
        "Naive Bayes":         MultinomialNB(alpha=0.4),
        "Random Forest":       RandomForestClassifier(n_estimators=CONFIG["rf_trees"],
                                                      max_depth=CONFIG["rf_depth"],
                                                      random_state=42),
        "Neural Network":      NeuralNetwork(hidden=CONFIG["nn_hidden"],
                                             epochs=CONFIG["nn_epochs"], lr=3e-3),
    }


def evaluate(name, model, X_tr, y_tr, X_te, y_te, buf, results, curves, save_cm=True):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_s = time.time() - t0

    t0 = time.time()
    pred = model.predict(X_te)
    pred_s = time.time() - t0
    proba = model.predict_proba(X_te)[:, 1]

    cm = M.confusion_matrix(y_te, pred)
    fpr, tpr = M.roc_curve(y_te, proba)
    auc = M.roc_auc(y_te, proba)

    results[name] = {
        "accuracy":  M.accuracy(y_te, pred),
        "precision": M.precision(y_te, pred),
        "recall":    M.recall(y_te, pred),
        "f1":        M.f1(y_te, pred),
        "specificity": M.specificity(y_te, pred),
        "roc_auc":   auc,
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "fit_seconds":     round(fit_s, 3),
        "predict_seconds": round(pred_s, 3),
        "confusion_matrix": cm.tolist(),
    }
    curves[name] = (fpr, tpr, auc)

    r = results[name]
    log(f"\n--- {name} ---", buf)
    log(f"  accuracy {r['accuracy']:.4f} | precision {r['precision']:.4f} | "
        f"recall {r['recall']:.4f} | F1 {r['f1']:.4f} | AUC {auc:.4f}", buf)
    log(f"  false positives {cm[0, 1]} (legit blocked) | "
        f"false negatives {cm[1, 0]} (phish delivered) | fit {fit_s:.2f}s", buf)
    log(M.classification_report(y_te, pred, LABELS), buf)

    if save_cm:
        M.plot_confusion_matrix(
            cm, ["LEGIT", "PHISH"], f"Confusion matrix - {name}",
            f"{RESULTS}/cm_{name.lower().replace(' ', '_')}.png", cmap="Oranges")
    return model


def ablation(views: dict, y_tr, y_te, buf):
    """Train every model on every feature view - the core comparative result."""
    rows = []
    log("\n" + "=" * 74, buf)
    log("[Phase 5] ABLATION - which feature channel carries the signal?", buf)
    log("=" * 74, buf)
    for view, (A_tr, A_te) in views.items():
        for name, model in make_models().items():
            model.fit(A_tr, y_tr)
            pred = model.predict(A_te)
            rows.append({"view": view, "model": name,
                         "accuracy": round(M.accuracy(y_te, pred), 4),
                         "f1": round(M.f1(y_te, pred), 4),
                         "recall": round(M.recall(y_te, pred), 4)})
            log(f"  {view:<22} {name:<22} acc {rows[-1]['accuracy']:.4f}  "
                f"F1 {rows[-1]['f1']:.4f}", buf)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{RESULTS}/ablation.csv", index=False)

    import matplotlib.pyplot as plt
    piv = tab.pivot(index="model", columns="view", values="f1")
    order = ["Text (TF-IDF) only", "Metadata only", "Text + Metadata"]
    piv = piv[[c for c in order if c in piv.columns]]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    x = np.arange(len(piv.index))
    w = 0.26
    for i, col in enumerate(piv.columns):
        bars = ax.bar(x + (i - 1) * w, piv[col], w, label=col,
                      color=M.PALETTE[i], edgecolor="white")
        for b, v in zip(bars, piv[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x, piv.index, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 score")
    ax.set_title("Feature-channel ablation", fontsize=12, weight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig5_ablation.png", dpi=170)
    plt.close(fig)
    return tab


def sklearn_crosscheck(X_tr, y_tr, X_te, y_te, buf):
    try:
        from sklearn.linear_model import LogisticRegression as SkLR
        from sklearn.ensemble import RandomForestClassifier as SkRF
        from sklearn.naive_bayes import MultinomialNB as SkNB
        from sklearn.neural_network import MLPClassifier as SkMLP
    except ImportError:
        log("\n[Cross-check] scikit-learn unavailable - skipped.", buf)
        return {}

    ref = {"Logistic Regression": SkLR(max_iter=1000),
           "Naive Bayes":         SkNB(alpha=0.4),
           "Random Forest":       SkRF(n_estimators=CONFIG["rf_trees"],
                                       max_depth=CONFIG["rf_depth"], random_state=42),
           "Neural Network":      SkMLP(hidden_layer_sizes=(CONFIG["nn_hidden"],),
                                        max_iter=300, random_state=42)}
    out = {}
    log("\n[Cross-check] scikit-learn reference accuracies (combined view)", buf)
    for name, m in ref.items():
        m.fit(X_tr, y_tr)
        acc = M.accuracy(y_te, m.predict(X_te))
        out[name] = acc
        log(f"  {name:<22} {acc:.4f}", buf)
    return out


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    buf: list[str] = []
    log("=" * 74, buf)
    log("AI-DRIVEN PHISHING EMAIL DETECTION USING NLP", buf)
    log("IICT Summer Internship 2026  |  Project 2", buf)
    log("=" * 74, buf)

    # ---- Phase 1-2 ---- #
    df = load_dataset(buf)
    df = clean_corpus(df, buf)

    counts = {"Legitimate": int((df.label == 0).sum()),
              "Phishing": int((df.label == 1).sum())}
    log(f"[Phase 2] Class balance: {counts}", buf)
    M.plot_class_balance(counts, f"{RESULTS}/fig1_class_balance.png",
                         "E-mail corpus class balance")

    # ---- Phase 3: features ---- #
    t0 = time.time()
    tokens = [preprocess(t, stem=True, drop_stopwords=True) for t in df["clean_text"]]
    log(f"\n[Phase 3] Tokenised {len(tokens)} messages in {time.time() - t0:.1f}s", buf)

    Xm_all, meta_names = F.build_matrix(df)
    log(f"[Phase 3] Structural feature matrix: {Xm_all.shape} "
        f"({len(meta_names)} hand-designed features)", buf)
    metadata_report(df, Xm_all, meta_names, buf)

    y = df["label"].to_numpy()
    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"], stratify=True)
    log(f"\n[Phase 3] Split: {len(idx_tr)} train / {len(idx_te)} test (stratified)", buf)

    tfidf = TfidfVectorizer(max_features=CONFIG["max_features"],
                            min_df=CONFIG["min_df"], ngram_range=CONFIG["ngram_range"])
    T_tr = tfidf.fit_transform([tokens[i] for i in idx_tr])
    T_te = tfidf.transform([tokens[i] for i in idx_te])
    text_names = tfidf.get_feature_names()
    log(f"[Phase 3] TF-IDF matrix: {T_tr.shape}, sparsity {100 * (T_tr == 0).mean():.1f}%", buf)

    scaler = F.MinMaxScaler().fit(Xm_all[idx_tr])
    Mm_tr, Mm_te = scaler.transform(Xm_all[idx_tr]), scaler.transform(Xm_all[idx_te])

    C_tr = np.hstack([T_tr, Mm_tr]).astype(np.float32)
    C_te = np.hstack([T_te, Mm_te]).astype(np.float32)
    all_names = text_names + meta_names
    log(f"[Phase 3] Combined matrix: {C_tr.shape}", buf)

    # ---- Phase 4-5: models on the combined view ---- #
    log("\n" + "=" * 74, buf)
    log("[Phase 4] TRAINING FOUR FROM-SCRATCH CLASSIFIERS (text + metadata)", buf)
    log("=" * 74, buf)

    results, curves, fitted = {}, {}, {}
    for name, model in make_models().items():
        fitted[name] = evaluate(name, model, C_tr, y_tr, C_te, y_te,
                                buf, results, curves)

    M.plot_model_comparison(results, f"{RESULTS}/fig3_model_comparison.png",
                            "Model comparison - phishing detection (text + metadata)")
    M.plot_roc_curves(curves, f"{RESULTS}/fig4_roc_curves.png",
                      "ROC curves - phishing e-mail detection")

    # ---- interpretability ---- #
    pos, neg = fitted["Logistic Regression"].top_features(all_names, k=14)
    M.plot_top_features(pos, neg, f"{RESULTS}/fig6_top_features.png",
                        "Most influential features (logistic regression)",
                        "Pushes toward PHISHING", "Pushes toward LEGITIMATE")
    log(f"\n[Phase 5] Strongest phishing cues   : {[w for w, _ in pos[:8]]}", buf)
    log(f"[Phase 5] Strongest legitimate cues : {[w for w, _ in neg[:8]]}", buf)

    imp = fitted["Random Forest"].feature_importances_()
    top_rf = np.argsort(-imp)[:12]
    log(f"[Phase 5] Random-forest top splits  : {[all_names[i] for i in top_rf]}", buf)

    import matplotlib.pyplot as plt
    sel = top_rf[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.barh(range(len(sel)), imp[sel], color="#17A398")
    ax.set_yticks(range(len(sel)), [all_names[i] for i in sel], fontsize=8)
    ax.set_xlabel("Split-frequency importance")
    ax.set_title("Random forest - most used features", fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig7_rf_importance.png", dpi=170)
    plt.close(fig)

    # ---- ablation ---- #
    abl = ablation({"Text (TF-IDF) only": (T_tr, T_te),
                    "Metadata only":      (Mm_tr, Mm_te),
                    "Text + Metadata":    (C_tr, C_te)},
                   y_tr, y_te, buf)

    # ---- cross-validation + cross-check ---- #
    scores = []
    for tr, va in M.k_fold_indices(y_tr, k=CONFIG["cv_folds"],
                                   random_state=CONFIG["random_state"]):
        m = LogisticRegression(lr=1.5, n_iters=1200).fit(C_tr[tr], y_tr[tr])
        scores.append(M.f1(y_tr[va], m.predict(C_tr[va])))
    scores = np.array(scores)
    log(f"\n[Phase 5] {CONFIG['cv_folds']}-fold CV F1 (Logistic Regression): "
        f"{scores.mean():.4f} +/- {scores.std():.4f}", buf)

    sk = sklearn_crosscheck(C_tr, y_tr, C_te, y_te, buf)

    best = max(results, key=lambda k: results[k]["f1"])
    log(f"\n[Phase 5] BEST MODEL BY F1 : {best} "
        f"(F1 = {results[best]['f1']:.4f}, recall = {results[best]['recall']:.4f})", buf)

    # ---- persist ---- #
    table = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc",
         "false_positives", "false_negatives", "fit_seconds", "predict_seconds"]]
    table.to_csv(f"{RESULTS}/metrics.csv")
    with open(f"{RESULTS}/metrics.json", "w") as f:
        json.dump({"config": {k: list(v) if isinstance(v, tuple) else v
                              for k, v in CONFIG.items()},
                   "dataset": {"n": int(len(df)), **counts,
                               "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
                               "vocab": int(len(text_names)),
                               "metadata_features": len(meta_names)},
                   "results": results,
                   "ablation": abl.to_dict(orient="records"),
                   "cv_f1_mean": float(scores.mean()), "cv_f1_std": float(scores.std()),
                   "sklearn_reference": sk,
                   "best_model": best,
                   "top_phish_features": [w for w, _ in pos[:12]],
                   "top_legit_features": [w for w, _ in neg[:12]],
                   "rf_top_features": [all_names[i] for i in top_rf]},
                  f, indent=2)

    log("\n" + table.round(4).to_string(), buf)
    with open(f"{RESULTS}/summary.txt", "w") as f:
        f.write("\n".join(buf))
    print(f"\nSaved figures + metrics to ./{RESULTS}/")


if __name__ == "__main__":
    main()
