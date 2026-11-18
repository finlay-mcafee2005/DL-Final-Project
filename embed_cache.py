"""Compute and save speech and resolution embeddings (run once).

Reads data/processed/ (built by un_data.py) and writes, per encoder, to
data/embeddings/<model-slug>/:

    speeches.npy      float32 (n_speeches, dim), one row per speech_id
    speech_ids.npy    speech_id for each row, same order as speeches.csv
    resolutions.npy   float32 (n_resolutions, dim), one row per undl_id
    undl_ids.npy      undl_id for each row, same order as resolutions.csv
    meta.json         encoder name, dim, chunking settings, row counts

Speeches: a General Debate speech runs ~2,900 words, far past the encoder's token
limit (256 for all-MiniLM-L6-v2), so each speech is split into chunks of whole
sentences up to CHUNK_WORDS words. Each chunk is embedded, the chunk vectors are
mean-pooled into one vector per speech, and that vector is L2-normalised.

Resolutions: the title with its boilerplate removed (": resolution / adopted by the
General Assembly"), plus the agenda item and subject tags when they add something.

Every vector is unit length, so a dot product is cosine similarity.

Usage:
    uv run python embed_cache.py                       # default encoder
    uv run python embed_cache.py --model all-mpnet-base-v2
    uv run python embed_cache.py --limit 50            # quick smoke test
    from embed_cache import load_embeddings
    ids, X = load_embeddings("speeches")               # or "resolutions"
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from un_data import DATA, load_processed

EMB = DATA / "embeddings"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# ~150 words is ~200 word-piece tokens, which leaves headroom under MiniLM's 256
# limit so chunks are not silently truncated by the encoder.
CHUNK_WORDS = 150

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------- text prep

def chunk_text(text, max_words=CHUNK_WORDS):
    """Split text into chunks of whole sentences, each at most max_words words.

    A single sentence longer than max_words is cut into max_words-word pieces.
    Every word of the text ends up in exactly one chunk, in order.
    """
    chunks, current = [], []
    for sentence in _SENTENCE_END.split(text):
        words = sentence.split()
        while len(words) > max_words:  # overly long sentence: hard split
            if current:
                chunks.append(" ".join(current))
                current = []
            chunks.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if current and len(current) + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
        current += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _clean(s):
    return s.fillna("").str.strip().str.rstrip(".")


def resolution_text(resolutions):
    """One short description per resolution: title, agenda item, subject tags."""
    title = _clean(resolutions["title"].str.split(" : ").str[0])
    agenda = _clean(resolutions["agenda_title"])
    subjects = _clean(resolutions["subjects"]).str.replace("--", ", ").str.lower()
    text = title.copy()
    add_agenda = (agenda != "") & (agenda.str.lower() != title.str.lower())
    text[add_agenda] += ". Agenda: " + agenda[add_agenda]
    text[subjects != ""] += ". Subjects: " + subjects[subjects != ""]
    return text + "."


# ---------------------------------------------------------------- encoding

def load_encoder(name, device=None):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


def embed_speeches(texts, model, batch_size=64, max_words=CHUNK_WORDS):
    """Mean-pooled, L2-normalised embedding for each (long) text."""
    chunks, owner = [], []
    for i, text in enumerate(texts):
        c = chunk_text(text, max_words)
        chunks += c
        owner += [i] * len(c)
    print(f"  {len(texts):,} speeches -> {len(chunks):,} chunks")
    X = model.encode(chunks, batch_size=batch_size, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
    owner = np.asarray(owner)
    pooled = np.zeros((len(texts), X.shape[1]), dtype=np.float64)
    np.add.at(pooled, owner, X)
    pooled /= np.bincount(owner, minlength=len(texts))[:, None]
    return _normalize(pooled)


def embed_short(texts, model, batch_size=64):
    X = model.encode(list(texts), batch_size=batch_size, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
    return _normalize(X)


def _normalize(X):
    X = np.asarray(X, dtype=np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


# ---------------------------------------------------------------- cache on disk

def model_slug(name):
    return name.split("/")[-1]


def cache_dir(model=DEFAULT_MODEL):
    return EMB / model_slug(model)


def build(model_name=DEFAULT_MODEL, limit=None, batch_size=64, device=None, model=None):
    """Embed every speech and resolution and save them. Returns the output folder.

    `model` can be passed in directly (e.g. a stand-in encoder in tests); otherwise
    the sentence-transformer `model_name` is loaded.
    """
    speeches, _, resolutions = load_processed()
    if limit:
        speeches, resolutions = speeches.head(limit), resolutions.head(limit)
    if model is None:
        model = load_encoder(model_name, device)

    t0 = time.time()
    print("embedding resolutions")
    R = embed_short(resolution_text(resolutions), model, batch_size)
    print("embedding speeches")
    S = embed_speeches(speeches["text"].tolist(), model, batch_size)

    out = cache_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "speeches.npy", S)
    np.save(out / "speech_ids.npy", speeches["speech_id"].to_numpy(dtype=str))
    np.save(out / "resolutions.npy", R)
    np.save(out / "undl_ids.npy", resolutions["undl_id"].to_numpy(dtype=np.int64))
    meta = {"model": model_name, "dim": int(S.shape[1]), "chunk_words": CHUNK_WORDS,
            "pooling": "mean of normalised chunk embeddings, then L2-normalised",
            "n_speeches": len(S), "n_resolutions": len(R), "limit": limit,
            "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


def load_embeddings(kind, model=DEFAULT_MODEL):
    """Return (ids, X) for kind "speeches" (ids = speech_id) or "resolutions" (undl_id)."""
    d = cache_dir(model)
    if kind == "speeches":
        return np.load(d / "speech_ids.npy"), np.load(d / "speeches.npy")
    if kind == "resolutions":
        return np.load(d / "undl_ids.npy"), np.load(d / "resolutions.npy")
    raise ValueError(f"unknown kind {kind!r}")


def lookup(kind, keys, model=DEFAULT_MODEL):
    """Embedding rows in the order of `keys` (e.g. votes["speech_id"]).

    Handy for building model inputs straight from a load_split() frame:
        S = lookup("speeches", votes["speech_id"])
        R = lookup("resolutions", votes["undl_id"])
    """
    ids, X = load_embeddings(kind, model)
    row = pd.Index(ids).get_indexer(np.asarray(keys))
    if (row < 0).any():
        raise KeyError(f"{(row < 0).sum()} {kind} have no cached embedding")
    return X[row]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, help="embed only the first N rows of each (smoke test)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", help="cpu, cuda or mps (default: auto)")
    a = p.parse_args()
    out = build(a.model, a.limit, a.batch_size, a.device)
    print(json.dumps(json.loads((out / "meta.json").read_text()), indent=2))
    print(f"written to {out}")
