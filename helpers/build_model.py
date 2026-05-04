# helpers/two_stage_top_words.py

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def top_words_binary_logreg(clf, vectorizer, top_n=15):
    """
    Returns top words pushing toward each class for binary LogisticRegression.
    In sklearn binary LR: coef_[0] corresponds to classes_[1].
      +weights -> classes_[1]
      -weights -> classes_[0]
    """
    feats = np.array(vectorizer.get_feature_names_out())
    weights = clf.coef_[0]
    cls0, cls1 = clf.classes_

    top_cls1 = feats[np.argsort(weights)[-top_n:]][::-1]
    top_cls0 = feats[np.argsort(weights)[:top_n]]

    return {cls0: list(top_cls0), cls1: list(top_cls1)}


def fit_binary_logreg_with_top_words(
    X,
    y,
    *,
    top_n=15,
    test_size=0.2,
    random_state=42,
    max_features=20000,
):
    """Fit TF–IDF + balanced LogReg and return top words by class."""
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_tr_t = tfidf.fit_transform(X_tr)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_tr_t, y_tr)

    return top_words_binary_logreg(clf, tfidf, top_n=top_n)


def extract_two_stage_top_words_by_platform(
    df,
    *,
    text_col,
    label_col,
    top_n=15,
):
    """
    Returns:
      dict[platform] -> {
        "stage1": {class: [words]},
        "stage2": {class: [words]} | None
      }
    """
    platform_top_words = {}

    for platform, sub in df.groupby("platform"):
        sub = sub.dropna(subset=[text_col, label_col]).copy()
        if sub.empty:
            continue

        # ---------- Stage 1: neutral vs nonneutral ----------
        sub["label_stage1"] = sub[label_col].apply(
            lambda x: "nonneutral" if x in ["positive", "negative"] else "neutral"
        )

        X1 = sub[text_col].astype(str).values
        y1 = sub["label_stage1"].values

        if len(np.unique(y1)) < 2:
            continue

        stage1_words = fit_binary_logreg_with_top_words(
            X1, y1, top_n=top_n
        )

        # ---------- Stage 2: positive vs negative ----------
        sub2 = sub[sub[label_col] != "neutral"]
        X2 = sub2[text_col].astype(str).values
        y2 = sub2[label_col].values

        if len(np.unique(y2)) < 2:
            stage2_words = None
        else:
            stage2_words = fit_binary_logreg_with_top_words(
                X2, y2, top_n=top_n
            )

        platform_top_words[platform] = {
            "stage1": stage1_words,
            "stage2": stage2_words,
        }

    return platform_top_words
