"""Tests for the UN speeches/votes project.

Run with: uv run pytest test_un.py -v
Requires data/processed/ (build it with `uv run python un_data.py`).
"""

import json

import numpy as np
import pytest

from comparison import EVAL_SPLITS, always_yes, best_threshold, last_session_vote, scores, target
from embed_cache import DEFAULT_MODEL, cache_dir, chunk_text, load_embeddings
from un_data import OUT, TEST_START, VAL_START, load_processed, load_split

pytestmark = pytest.mark.skipif(not (OUT / "speeches.csv").exists(),
                                reason="run `uv run python un_data.py` first")


@pytest.fixture(scope="module")
def data():
    return load_processed()


@pytest.fixture(scope="module")
def splits():
    return {s: load_split(s) for s in ["train", "val", "test", "test_heldout"]}


# ---------------------------------------------------------------- data: shapes, no NaNs

def test_one_speech_per_country_session(data):
    speeches, _, _ = data
    assert speeches["speech_id"].is_unique
    assert not speeches.duplicated(["iso3", "session"]).any()


def test_no_nans_in_required_columns(data):
    speeches, votes, resolutions = data
    assert not speeches[["speech_id", "iso3", "session", "year", "split", "text"]].isna().any().any()
    assert not votes[["speech_id", "undl_id", "vote"]].isna().any().any()
    assert not resolutions[["undl_id", "session", "title", "contested"]].isna().any().any()


def test_every_speech_has_votes_and_every_vote_a_speech(data):
    speeches, votes, resolutions = data
    assert set(speeches["speech_id"]) == set(votes["speech_id"])
    assert set(votes["undl_id"]) <= set(resolutions["undl_id"])
    assert (speeches["n_votes"] == speeches["speech_id"].map(votes.groupby("speech_id").size())).all()


def test_votes_are_yes_no_abstain(data):
    _, votes, _ = data
    assert set(votes["vote"]) == {"Y", "N", "A"}
    assert set(votes["last_vote"].dropna()) <= {"Y", "N", "A"}
    assert not votes.duplicated(["speech_id", "undl_id"]).any()


def test_votes_are_from_the_speech_session(data):
    speeches, votes, resolutions = data
    v = votes.merge(resolutions[["undl_id", "session"]], on="undl_id")
    v = v.merge(speeches[["speech_id", "session"]], on="speech_id", suffixes=("_vote", "_speech"))
    assert (v["session_vote"] == v["session_speech"]).all()


# ---------------------------------------------------------------- no time leakage

def test_splits_are_ordered_in_time(splits):
    train, val, test = (splits[s][0]["year"] for s in ["train", "val", "test"])
    assert train.max() < VAL_START <= val.min()
    assert val.max() < TEST_START <= test.min()
    assert splits["test_heldout"][0]["year"].min() >= TEST_START


def test_heldout_countries_never_in_train_or_val(splits):
    heldout = set(splits["test_heldout"][0]["iso3"])
    assert heldout
    for s in ["train", "val", "test"]:
        assert not heldout & set(splits[s][0]["iso3"])


def test_splits_do_not_share_speeches(splits):
    ids = [set(splits[s][0]["speech_id"]) for s in splits]
    assert sum(map(len, ids)) == len(set().union(*ids))


def test_history_only_uses_earlier_sessions(splits):
    for _, votes in splits.values():
        h = votes.dropna(subset=["last_vote_session"])
        assert (h["last_vote_session"] < h["session"]).all()


# ---------------------------------------------------------------- embeddings
# Built with `uv run python embed_cache.py`; skipped until the cache exists.

needs_embeddings = pytest.mark.skipif(
    not (cache_dir(DEFAULT_MODEL) / "meta.json").exists(),
    reason="run `uv run python embed_cache.py` first")


def test_chunks_keep_every_word_in_order_and_respect_the_limit():
    text = "Short one. " + "word " * 400 + "end. Another sentence here! And one more?"
    chunks = chunk_text(text, max_words=50)
    assert " ".join(chunks).split() == text.split()
    assert all(0 < len(c.split()) <= 50 for c in chunks)


@needs_embeddings
@pytest.mark.parametrize("kind,key", [("speeches", "speech_id"), ("resolutions", "undl_id")])
def test_every_row_has_one_finite_unit_embedding(data, kind, key):
    speeches, _, resolutions = data
    ids, X = load_embeddings(kind)
    meta = json.loads((cache_dir(DEFAULT_MODEL) / "meta.json").read_text())
    expected = (speeches if kind == "speeches" else resolutions)[key]
    if meta["limit"] is None:  # a --limit smoke-test cache only covers the first rows
        assert set(ids) == set(expected)
    assert len(ids) == len(set(ids)) == len(X)
    assert X.shape[1] == meta["dim"]
    assert np.isfinite(X).all()
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-4)


# ---------------------------------------------------------------- baselines

def test_scores_on_a_known_case():
    y = np.array([1, 1, 1, 0])
    assert scores(y, np.ones(4))["accuracy"] == 0.75
    assert scores(y, np.ones(4))["balanced_acc"] == 0.5
    assert scores(y, y.astype(float))["macro_f1"] == 1.0
    assert scores(y, np.array([0.9, 0.8, 0.7, 0.6]), best_threshold(y, np.array([0.9, 0.8, 0.7, 0.6])))["macro_f1"] == 1.0


@pytest.mark.parametrize("baseline", [always_yes, last_session_vote])
def test_baselines_give_one_probability_per_vote(splits, baseline):
    probs, threshold = baseline(splits)
    for s in EVAL_SPLITS:
        assert len(probs[s]) == len(splits[s][1])
        assert ((probs[s] >= 0) & (probs[s] <= 1)).all()
    assert 0 <= threshold <= 1


def test_last_session_vote_beats_always_yes_on_contested_test(splits):
    votes = splits["test"][1]
    contested = votes["contested"].to_numpy(bool)
    y = target(votes)[contested]
    history = last_session_vote(splits)[0]["test"][contested]
    assert scores(y, history)["macro_f1"] > scores(y, np.ones(len(y)))["macro_f1"] + 0.2
