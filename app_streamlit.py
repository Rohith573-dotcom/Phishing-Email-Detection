"""
app_streamlit.py
================
Optional deployment deliverable for Project 2 - a small web interface that
classifies a pasted e-mail as PHISHING or LEGITIMATE and shows *why*.

Run
---
    pip install streamlit
    streamlit run app_streamlit.py

The model is trained once on first load and cached, so the page is responsive
after the initial few seconds.  Everything it uses is the from-scratch code in
`mlcore/` and `features.py` - Streamlit only supplies the widgets.

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlcore.preprocessing import preprocess, train_test_split
from mlcore.vectorizers import TfidfVectorizer
from mlcore.models import LogisticRegression
from mlcore import metrics as M
import features as F

st.set_page_config(page_title="Phishing E-mail Detector", page_icon="🛡️",
                   layout="centered")


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training the detector (first run only)...")
def load_model():
    path = "data/emails.csv" if os.path.exists("data/emails.csv") else "data/emails_raw.csv"
    df = pd.read_csv(path).dropna(subset=["body", "label"])
    df["label"] = df["label"].astype(int)
    df["clean_text"] = (df["subject"].astype(str).map(F.clean_email_body) + " " +
                        df["body"].astype(str).map(F.clean_email_body))

    tokens = [preprocess(t) for t in df["clean_text"]]
    Xm, meta_names = F.build_matrix(df)
    y = df["label"].to_numpy()

    idx = np.arange(len(y))
    itr, ite, ytr, yte = train_test_split(idx, y, test_size=0.2, random_state=42)

    tfidf = TfidfVectorizer(max_features=1200, min_df=3, ngram_range=(1, 2))
    T_tr = tfidf.fit_transform([tokens[i] for i in itr])
    T_te = tfidf.transform([tokens[i] for i in ite])

    scaler = F.MinMaxScaler().fit(Xm[itr])
    C_tr = np.hstack([T_tr, scaler.transform(Xm[itr])]).astype(np.float32)
    C_te = np.hstack([T_te, scaler.transform(Xm[ite])]).astype(np.float32)

    model = LogisticRegression(lr=0.6, n_iters=700).fit(C_tr, ytr)
    pred = model.predict(C_te)
    stats = {"accuracy": M.accuracy(yte, pred), "recall": M.recall(yte, pred),
             "precision": M.precision(yte, pred), "f1": M.f1(yte, pred),
             "n_train": len(itr), "source": path}
    names = tfidf.get_feature_names() + meta_names
    return model, tfidf, scaler, names, stats


def score(sender, subject, body, model, tfidf, scaler):
    text = F.clean_email_body(subject) + " " + F.clean_email_body(body)
    t = tfidf.transform([preprocess(text)])
    raw = F.extract_features(sender, subject, body)
    m = scaler.transform(np.array([[raw[k] for k in F.FEATURE_NAMES]], dtype=np.float32))
    x = np.hstack([t, m]).astype(np.float32)
    return float(model.predict_proba(x)[0, 1]), x, raw


# --------------------------------------------------------------------------- #
st.title("🛡️ Phishing E-mail Detector")
st.caption("Project 2 · IICT Summer Internship 2026 · AI & ML — "
           "a defensive classifier built from scratch in NumPy")

model, tfidf, scaler, names, stats = load_model()

with st.sidebar:
    st.subheader("Model")
    st.write("Logistic regression on TF-IDF + 17 structural features")
    st.metric("Test accuracy", f"{stats['accuracy']:.1%}")
    st.metric("Recall (phish caught)", f"{stats['recall']:.1%}")
    st.metric("Precision", f"{stats['precision']:.1%}")
    st.caption(f"Trained on {stats['n_train']} messages from `{stats['source']}`")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Lower catches more phishing but quarantines more "
                               "legitimate mail.")

EXAMPLES = {
    "— paste your own —": ("", "", ""),
    "Suspicious account notice": (
        "security@account-support.example.net",
        "URGENT: Your account will be suspended",
        "Dear Customer, Your account has been temporarily suspended due to unusual "
        "sign in activity. You must verify your identity within 24 hours or access "
        "will be permanently revoked. Click the secure link below to confirm your "
        "login credentials immediately. http://203.0.113.24/secure/login/verify.php !!"),
    "Routine internal mail": (
        "priya.sharma@corp.example.com",
        "Minutes from the project review",
        "Hi team, Please find attached the minutes from yesterday's project review "
        "meeting. The sprint retrospective has been moved to Thursday at eleven. "
        "Let me know if Tuesday afternoon works for the client walkthrough. "
        "https://corp.example.com/wiki/minutes"),
}

choice = st.selectbox("Load an example", list(EXAMPLES))
ex_sender, ex_subject, ex_body = EXAMPLES[choice]

sender = st.text_input("From", value=ex_sender, placeholder="sender@domain.com")
subject = st.text_input("Subject", value=ex_subject)
body = st.text_area("Body", value=ex_body, height=200,
                    placeholder="Paste the full message text here...")

if st.button("Analyse message", type="primary", use_container_width=True):
    if not body.strip():
        st.warning("Paste a message body first.")
    else:
        p, x, raw = score(sender, subject, body, model, tfidf, scaler)
        verdict = "PHISHING" if p >= threshold else "LEGITIMATE"

        if verdict == "PHISHING":
            st.error(f"### ⚠️ {verdict}\nConfidence: **{p:.1%}**")
        else:
            st.success(f"### ✅ {verdict}\nPhishing probability: **{p:.1%}**")
        st.progress(min(max(p, 0.0), 1.0))

        # --- why: the features that moved this decision most --------------- #
        contrib = model.w * x[0]
        order = np.argsort(-np.abs(contrib))[:10]
        expl = pd.DataFrame({
            "feature": [names[i] for i in order],
            "contribution": np.round(contrib[order], 4),
            "pushes toward": ["phishing" if contrib[i] > 0 else "legitimate"
                              for i in order],
        })
        st.subheader("Why this verdict")
        st.dataframe(expl, use_container_width=True, hide_index=True)

        st.subheader("Structural signals")
        flags = {k: v for k, v in raw.items() if v}
        cols = st.columns(3)
        for i, (k, v) in enumerate(sorted(flags.items())):
            cols[i % 3].metric(k.replace("_", " "), f"{v:g}")

        st.caption("Quarantine rather than delete: a false positive that silently "
                   "destroys a legitimate message is worse than one that delays it.")

st.divider()
st.caption("Trained on a synthetic corpus for teaching purposes. Replace "
           "`data/emails.csv` with a real labelled corpus before operational use.")
