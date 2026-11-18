"""Classical baselines for predicting UN votes.

Every baseline predicts P(Yes) for each (country-session, resolution) vote, the same
target as the Stage 2 head (sigmoid + BCELoss), and is scored on val, test and
test_heldout, over all votes and over contested votes only.

    always_yes          P(Yes) = 1. Shows how misleading raw accuracy is.
    last_session_vote   The country's vote on the matching resolution last session
                        (Yes when there is no match). The strong baseline.
    tfidf_logreg        Logistic regression on TF-IDF of the speech and the resolution
                        text, reduced with SVD. No embeddings, no history.
    random_forest       Random forest on the cached sentence embeddings. No history.

The learned baselines see each pair as [speech, resolution, speech * resolution]: the
element-wise product lets a linear model pick up "this speech is about the same issue
as this resolution", which a plain concatenation cannot.

Thresholds: a model that outputs P(Yes) still needs a cut-off to make a Yes/No call,
and with ~82% Yes votes 0.5 is rarely the right one. Each learned baseline picks the
threshold that maximises macro-F1 on val, then applies it unchanged to test, so test
numbers stay honest (val numbers are optimistic for those two baselines).

Metrics: macro-F1 and balanced accuracy (the headline numbers), plus accuracy.

Usage:
    uv run python comparison.py                         # all baselines
    uv run python comparison.py --baseline last_session_vote
    uv run python comparison.py --rf-sample 50000       # faster random forest
Writes results/baselines.csv.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from embed_cache import lookup, resolution_text
from un_data import SEED, load_processed, load_split

RESULTS = Path(__file__).parent / "results"
EVAL_SPLITS = ["val", "test", "test_heldout"]


# ---------------------------------------------------------------- data

def target(votes):
    """1 when the country voted Yes, 0 for No or Abstain."""
    return (votes["vote"] == "Y").to_numpy(dtype=np.int8)


def load_all():
    return {s: load_split(s) for s in ["train", *EVAL_SPLITS]}


def pair_features(S, R):
    """[speech, resolution, speech * resolution] for each vote."""
    return np.hstack([S, R, S * R]).astype(np.float32)


# ---------------------------------------------------------------- metrics

def scores(y, p, threshold=0.5):
    pred = (p >= threshold).astype(np.int8)
    return {"macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_acc": balanced_accuracy_score(y, pred),
            "accuracy": accuracy_score(y, pred)}


def best_threshold(y, p):
    """Threshold on P(Yes) that maximises macro-F1 (searched on val)."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    return float(max(grid, key=lambda t: scores(y, p, t)["macro_f1"]))


# ---------------------------------------------------------------- baselines
# Each baseline takes the dict of splits and returns {split: P(Yes) per vote} for
# every split in EVAL_SPLITS, plus "threshold" (fixed 0.5 or tuned on val).

def always_yes(data, **_):
    return {s: np.ones(len(data[s][1])) for s in EVAL_SPLITS}, 0.5


def last_session_vote(data, **_):
    def predict(votes):
        return votes["last_vote"].fillna("Y").eq("Y").to_numpy(dtype=float)
    return {s: predict(data[s][1]) for s in EVAL_SPLITS}, 0.5


def tfidf_logreg(data, svd_dim=128, **_):
    speeches, _, resolutions = load_processed()
    train_sp = data["train"][0]
    train_res = resolutions[resolutions["session"].isin(train_sp["session"])]
    res_text = resolution_text(resolutions)

    # Fit vocabulary and SVD on training text only, then project everything.
    def projector(train_text, all_text, max_features):
        tfidf = TfidfVectorizer(max_features=max_features, min_df=5, max_df=0.9,
                                stop_words="english", sublinear_tf=True)
        svd = TruncatedSVD(svd_dim, random_state=SEED)
        svd.fit(tfidf.fit_transform(train_text))
        X = svd.transform(tfidf.transform(all_text))
        return X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-8)

    S_all = projector(train_sp["text"], speeches["text"], 50_000)
    R_all = projector(res_text[train_res.index], res_text, 20_000)
    s_row = pd.Series(np.arange(len(speeches)), index=speeches["speech_id"])
    r_row = pd.Series(np.arange(len(resolutions)), index=resolutions["undl_id"])

    def X(votes):
        return pair_features(S_all[s_row[votes["speech_id"]].to_numpy()],
                             R_all[r_row[votes["undl_id"]].to_numpy()])

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X(data["train"][1]))
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(Xtr, target(data["train"][1]))
    probs = {s: model.predict_proba(scaler.transform(X(data[s][1])))[:, 1] for s in EVAL_SPLITS}
    return probs, best_threshold(target(data["val"][1]), probs["val"])


def random_forest(data, rf_sample=150_000, **_):
    def X(votes):
        return pair_features(lookup("speeches", votes["speech_id"]),
                             lookup("resolutions", votes["undl_id"]))

    train = data["train"][1]
    if rf_sample and len(train) > rf_sample:  # full train set is ~534k rows x 1152 features
        train = train.sample(rf_sample, random_state=SEED)
    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=20,
                                   max_features="sqrt", n_jobs=-1, random_state=SEED)
    model.fit(X(train), target(train))
    probs = {s: model.predict_proba(X(data[s][1]))[:, 1] for s in EVAL_SPLITS}
    return probs, best_threshold(target(data["val"][1]), probs["val"])


BASELINES = {f.__name__: f for f in [always_yes, last_session_vote, tfidf_logreg, random_forest]}


# ---------------------------------------------------------------- run

def evaluate(name, data, probs, threshold, seconds):
    rows = []
    for split in EVAL_SPLITS:
        votes = data[split][1]
        y, p = target(votes), probs[split]
        for subset, mask in [("all", np.ones(len(votes), bool)),
                             ("contested", votes["contested"].to_numpy(bool))]:
            rows.append({"baseline": name, "split": split, "subset": subset,
                         "n": int(mask.sum()), **scores(y[mask], p[mask], threshold),
                         "threshold": round(threshold, 4), "seconds": round(seconds, 1)})
    return rows


def run(names, **kwargs):
    data = load_all()
    rows = []
    for name in names:
        print(f"running {name}")
        t0 = time.time()
        probs, threshold = BASELINES[name](data, **kwargs)
        rows += evaluate(name, data, probs, threshold, time.time() - t0)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--baseline", choices=list(BASELINES), action="append",
                   help="run only this baseline (repeatable); default: all")
    p.add_argument("--rf-sample", type=int, default=150_000,
                   help="training rows for random_forest (0 = all)")
    a = p.parse_args()

    names = a.baseline or list(BASELINES)
    results = run(names, rf_sample=a.rf_sample)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "baselines.csv"
    if a.baseline and out.exists():  # keep earlier results for baselines not re-run
        old = pd.read_csv(out)
        results = pd.concat([old[~old["baseline"].isin(names)], results], ignore_index=True)
    results.to_csv(out, index=False)

    table = results.pivot_table(index="baseline", columns=["split", "subset"],
                                values="macro_f1", sort=False).round(3)
    print("\nmacro-F1 (threshold tuned on val for the learned baselines)")
    print(table.to_string())
    print(f"\nwritten to {out}")
