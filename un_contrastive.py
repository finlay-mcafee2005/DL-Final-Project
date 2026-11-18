"""Experiment functions, the shared train() loop, and logging.

Each experiment is a function that returns (model, loss_fn, optimizer), built from
nn.Sequential, and every one runs through the same train() loop (the Lab 3 setup).

Stage 2, vote prediction. The model reads one (country-session, resolution) pair and
outputs P(Yes) through a final sigmoid, trained with nn.BCELoss. Inputs come from:

    speech       cached speech embedding (frozen sentence encoder, 384 numbers)
    resolution   cached resolution embedding (384 numbers)
    history      the country's vote on the matching resolution last session,
                 one-hot over Yes / No / Abstain / no match (4 numbers)

Experiments (each changes one thing from another):

    probe_no_contrastive   speech + resolution          raw frozen embeddings, no Stage 1
    history_mlp            resolution + history         no speech: history alone, learned
    probe_plus_history     speech + resolution + history

probe_plus_history vs history_mlp is the main question in miniature: does the speech
add anything once the model already knows last session's vote? Stage 1 (contrastive)
experiments plug in here too, through the same train() loop.

Training: Adam, batches of 1,024 votes, up to --epochs epochs, keeping the weights
from the epoch with the best val macro-F1 (early stopping with --patience). As in
comparison.py, the Yes/No cut-off is tuned on val and then applied unchanged to test.

Usage:
    uv run python un_contrastive.py                                   # every experiment
    uv run python un_contrastive.py --experiment probe_no_contrastive # just one
    uv run python un_contrastive.py --epochs 5                        # quicker
Writes results/<experiment>.csv (per-epoch history) and results/summary.csv (final
scores, same columns as results/baselines.csv).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from comparison import EVAL_SPLITS, best_threshold, scores, target
from embed_cache import load_embeddings
from un_data import SEED, load_split

RESULTS = Path(__file__).parent / "results"
HISTORY_CODES = ["Y", "N", "A"]  # plus a fourth "no match" column


# ---------------------------------------------------------------- data

class VoteData:
    """One split as index arrays into the embedding tables, plus history and target.

    Embeddings are gathered per batch rather than copied per vote: the train split has
    534k votes, and a full copy of their features would take ~1.6 GB.
    """

    def __init__(self, votes, tables):
        S, s_row, R, r_row = tables
        self.S, self.R = S, R
        self.speech = torch.as_tensor(s_row.get_indexer(votes["speech_id"]), dtype=torch.long)
        self.res = torch.as_tensor(r_row.get_indexer(votes["undl_id"]), dtype=torch.long)
        if (self.speech < 0).any() or (self.res < 0).any():
            raise KeyError("some votes have no cached embedding; run embed_cache.py")
        self.history = torch.as_tensor(history_onehot(votes["last_vote"]))
        self.y = torch.as_tensor(target(votes), dtype=torch.float32)
        self.contested = votes["contested"].to_numpy(bool)

    def __len__(self):
        return len(self.y)

    def features(self, idx, inputs):
        parts = {"speech": lambda: self.S[self.speech[idx]],
                 "resolution": lambda: self.R[self.res[idx]],
                 "history": lambda: self.history[idx]}
        return torch.cat([parts[name]() for name in inputs], dim=1)


def history_onehot(last_vote):
    """Yes / No / Abstain / no-match one-hot for the country's last-session vote."""
    codes = pd.Categorical(last_vote, categories=HISTORY_CODES).codes  # -1 = missing
    out = np.zeros((len(codes), len(HISTORY_CODES) + 1), dtype=np.float32)
    out[np.arange(len(codes)), np.where(codes < 0, len(HISTORY_CODES), codes)] = 1
    return out


def load_tables():
    s_ids, S = load_embeddings("speeches")
    r_ids, R = load_embeddings("resolutions")
    return (torch.as_tensor(S), pd.Index(s_ids), torch.as_tensor(R), pd.Index(r_ids))


def load_data():
    tables = load_tables()
    return {s: VoteData(load_split(s)[1], tables) for s in ["train", *EVAL_SPLITS]}


def input_dim(data, inputs):
    return data.features(torch.arange(1), inputs).shape[1]


# ---------------------------------------------------------------- experiments
# Each takes the input size and returns (model, loss_fn, optimizer). The INPUTS table
# below says which features each one reads.

def prediction_head(in_dim):
    """Stage 2 head: two ReLU layers, then a sigmoid giving P(Yes)."""
    return nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                         nn.Linear(256, 64), nn.ReLU(),
                         nn.Linear(64, 1), nn.Sigmoid(), nn.Flatten(0))


def probe_no_contrastive(in_dim):
    """Stage 2 on the raw frozen embeddings: does contrastive training add anything?"""
    model = prediction_head(in_dim)
    return model, nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


def history_mlp(in_dim):
    """Same head, resolution + last-session vote only: the learned history baseline."""
    model = prediction_head(in_dim)
    return model, nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


def probe_plus_history(in_dim):
    """Same head, speech + resolution + last-session vote."""
    model = prediction_head(in_dim)
    return model, nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


EXPERIMENTS = {
    "probe_no_contrastive": (probe_no_contrastive, ["speech", "resolution"]),
    "history_mlp": (history_mlp, ["resolution", "history"]),
    "probe_plus_history": (probe_plus_history, ["speech", "resolution", "history"]),
}


# ---------------------------------------------------------------- training

def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, data, inputs, device, batch_size=8192):
    model.eval()
    out = [model(data.features(idx, inputs).to(device)).cpu()
           for idx in torch.arange(len(data)).split(batch_size)]
    return torch.cat(out).numpy()


def train(model, loss_fn, optimizer, data, inputs, epochs=20, batch_size=1024,
          patience=3, device=None, log=print):
    """Shared training loop. Returns (best model, per-epoch history DataFrame).

    Each epoch: one pass over shuffled training votes, then val loss and val macro-F1
    (threshold tuned on val). Keeps the weights from the best val macro-F1 epoch and
    stops after `patience` epochs without improvement.
    """
    device = device or pick_device()
    model.to(device)
    train_data, val = data["train"], data["val"]
    gen = torch.Generator().manual_seed(SEED)
    history, best, best_state, stale = [], -1.0, None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total, n = 0.0, 0
        for idx in torch.randperm(len(train_data), generator=gen).split(batch_size):
            x = train_data.features(idx, inputs).to(device)
            y = train_data.y[idx].to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)

        p = predict(model, val, inputs, device)
        y_val = val.y.numpy()
        val_loss = loss_fn(torch.as_tensor(p), val.y).item()
        threshold = best_threshold(y_val, p)
        f1 = scores(y_val, p, threshold)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss,
                        "val_macro_f1": f1, "threshold": threshold,
                        "seconds": round(time.time() - t0, 1)})
        log(f"  epoch {epoch:2d}  train loss {total / n:.4f}  val loss {val_loss:.4f}"
            f"  val macro-F1 {f1:.3f}  ({time.time() - t0:.0f}s)")

        if f1 > best:
            best, stale = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log(f"  no val improvement for {patience} epochs, stopping")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def evaluate(name, model, data, inputs, device, seconds):
    """Final scores in the same layout as results/baselines.csv."""
    probs = {s: predict(model, data[s], inputs, device) for s in EVAL_SPLITS}
    threshold = best_threshold(data["val"].y.numpy(), probs["val"])
    rows = []
    for split in EVAL_SPLITS:
        d = data[split]
        y = d.y.numpy().astype(np.int8)
        for subset, mask in [("all", np.ones(len(d), bool)), ("contested", d.contested)]:
            rows.append({"baseline": name, "split": split, "subset": subset,
                         "n": int(mask.sum()), **scores(y[mask], probs[split][mask], threshold),
                         "threshold": round(threshold, 4), "seconds": round(seconds, 1)})
    return rows


def run(names, epochs=20, batch_size=1024, patience=3, device=None):
    torch.manual_seed(SEED)
    device = pick_device(device)
    print(f"device: {device}")
    data = load_data()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for name in names:
        fn, inputs = EXPERIMENTS[name]
        print(f"running {name}  (inputs: {', '.join(inputs)})")
        torch.manual_seed(SEED)
        t0 = time.time()
        model, loss_fn, optimizer = fn(input_dim(data["train"], inputs))
        model, hist = train(model, loss_fn, optimizer, data, inputs, epochs, batch_size,
                            patience, device)
        hist.to_csv(RESULTS / f"{name}.csv", index=False)
        rows += evaluate(name, model, data, inputs, device, time.time() - t0)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", choices=list(EXPERIMENTS), action="append",
                   help="run only this experiment (repeatable); default: all")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", help="cpu, cuda or mps (default: auto)")
    a = p.parse_args()

    names = a.experiment or list(EXPERIMENTS)
    results = run(names, a.epochs, a.batch_size, a.patience, a.device)
    out = RESULTS / "summary.csv"
    if out.exists():  # keep earlier results for experiments not re-run
        old = pd.read_csv(out)
        results = pd.concat([old[~old["baseline"].isin(names)], results], ignore_index=True)
    results.to_csv(out, index=False)

    table = results.pivot_table(index="baseline", columns=["split", "subset"],
                                values="macro_f1", sort=False).round(3)
    print("\nmacro-F1 (threshold tuned on val)")
    print(table.to_string())
    baselines = RESULTS / "baselines.csv"
    if baselines.exists():
        b = pd.read_csv(baselines).query("split == 'test' and subset == 'contested'")
        print("\nfor comparison, test contested macro-F1: " + ", ".join(
            f"{r.baseline} {r.macro_f1:.3f}" for r in b.itertuples()))
    print(f"\nwritten to {out}")
