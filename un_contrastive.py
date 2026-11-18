"""Experiment functions, the shared training loops, and logging.

Each experiment is a function that returns (model, loss_fn, optimizer), built from
nn.Sequential (the Lab 3 setup). Stage 1 experiments share train_contrastive(); Stage 2
experiments share train().

Stage 1, contrastive pretraining (CLIP-style). Two projection heads map a country's
speech embedding (384 numbers) and its vote record for the same session (24 numbers,
un_data.vote_records) into one 128-dimensional space. In a batch of B country-sessions
the B x B cosine-similarity matrix should peak on the diagonal; the loss is
nn.CrossEntropyLoss over that matrix in both directions. Every batch holds speeches
from a single session, so the model has to match on positions, not on era (Cold War
speeches vs. recent ones would otherwise be trivially easy to tell apart).

    contrastive_relu_adam         reference: 2-layer ReLU heads, Adam, temperature 0.07
    contrastive_gelu_adam         ReLU -> GELU
    contrastive_relu_sgd          Adam -> SGD with momentum
    contrastive_relu_adam_t0.05   temperature 0.07 -> 0.05
    contrastive_relu_adam_dropout dropout 0.2 in the heads

Scored by retrieval within each session: for each speech, is its own country's vote
record the closest (top-1) or among the 10 closest (top-10)? Chance is 1/n and 10/n
for a session of n countries, and is reported alongside.

Stage 2, vote prediction. A head reads one (country-session, resolution) pair and
outputs P(Yes) through a final sigmoid, trained with nn.BCELoss. Inputs:

    speech       cached speech embedding (frozen sentence encoder, 384 numbers)
    speech_proj  the speech passed through a trained, frozen Stage 1 speech head (128)
    resolution   cached resolution embedding (384 numbers)
    history      the country's vote on the matching resolution last session,
                 one-hot over Yes / No / Abstain / no match (4 numbers)

    probe_no_contrastive      speech + resolution
    history_mlp               resolution + history         (no speech)
    probe_plus_history        speech + resolution + history
    contrastive_probe         speech_proj + resolution
    contrastive_plus_history  speech_proj + resolution + history   (the headline model)

contrastive_* vs probe_* asks whether Stage 1 helps; *_plus_history vs history_mlp asks
whether the speech adds anything once last session's vote is known.

    hybrid_history_or_speech  history_mlp's decision where a last-session vote exists,
                              contrastive_probe's where it does not (no new training)

Stage 2 training: Adam, batches of 1,024 votes, keeping the epoch with the best val
macro-F1 (early stopping with --patience). As in comparison.py, the Yes/No cut-off is
tuned on val and then applied unchanged to test.

Usage:
    uv run python un_contrastive.py                                      # everything
    uv run python un_contrastive.py --seeds 5            # everything, 5 seeds, mean ± std
    uv run python un_contrastive.py --experiment contrastive_relu_adam   # just one
    uv run python un_contrastive.py --experiment contrastive_plus_history \\
        --stage1 contrastive_gelu_adam      # Stage 2 on top of a different Stage 1
Writes results/<experiment>.csv (per-epoch history, one block per seed),
results/summary.csv (Stage 2 scores per seed, same columns as results/baselines.csv
plus `seed`) and results/retrieval.csv (Stage 1 retrieval per seed).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from comparison import EVAL_SPLITS, best_threshold, scores, subsets, target
from embed_cache import load_embeddings
from un_data import SEED, load_split, vote_records

RESULTS = Path(__file__).parent / "results"
HISTORY_CODES = ["Y", "N", "A"]  # plus a fourth "no match" column
SPLITS = ["train", *EVAL_SPLITS]


# ---------------------------------------------------------------- data

class VoteData:
    """Stage 2: one split as index arrays into the embedding tables, plus history and
    target.

    Embeddings are gathered per batch rather than copied per vote: the train split has
    534k votes, and a full copy of their features would take ~1.6 GB.
    """

    def __init__(self, votes, tables):
        S, s_row, R, r_row = tables
        self.S, self.R, self.P = S, R, None  # P: Stage 1 projections, set by run()
        self.speech = torch.as_tensor(s_row.get_indexer(votes["speech_id"]), dtype=torch.long)
        self.res = torch.as_tensor(r_row.get_indexer(votes["undl_id"]), dtype=torch.long)
        if (self.speech < 0).any() or (self.res < 0).any():
            raise KeyError("some votes have no cached embedding; run embed_cache.py")
        self.history = torch.as_tensor(history_onehot(votes["last_vote"]))
        self.y = torch.as_tensor(target(votes), dtype=torch.float32)
        self.subsets = subsets(votes)  # all / contested / no_history masks

    def __len__(self):
        return len(self.y)

    def features(self, idx, inputs):
        parts = {"speech": lambda: self.S[self.speech[idx]],
                 "speech_proj": lambda: self.P[self.speech[idx]],
                 "resolution": lambda: self.R[self.res[idx]],
                 "history": lambda: self.history[idx]}
        return torch.cat([parts[name]() for name in inputs], dim=1)


class PairData:
    """Stage 1: each speech in a split paired with its vote record, grouped by session."""

    def __init__(self, speeches, votes, tables):
        S, s_row = tables[0], tables[1]
        V = vote_records(votes)
        self.S = S[torch.as_tensor(s_row.get_indexer(V.index), dtype=torch.long)]
        self.V = torch.tensor(V.to_numpy())
        session = speeches.set_index("speech_id").loc[V.index, "session"].to_numpy()
        self.sessions = [torch.as_tensor(np.flatnonzero(session == s)) for s in np.unique(session)]

    def __len__(self):
        return len(self.V)

    def batches(self, gen=None, max_batch=256):
        """Index batches, each from one session. Shuffled when a generator is given."""
        order = (torch.randperm(len(self.sessions), generator=gen).tolist() if gen is not None
                 else range(len(self.sessions)))
        for i in order:
            idx = self.sessions[i]
            if gen is not None:
                idx = idx[torch.randperm(len(idx), generator=gen)]
            for batch in idx.split(max_batch):
                if len(batch) > 1:  # a batch of one has nothing to contrast with
                    yield batch


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
    """(tables, {split: VoteData}, {split: PairData})."""
    tables = load_tables()
    splits = {s: load_split(s) for s in SPLITS}
    votes = {s: VoteData(v, tables) for s, (_, v) in splits.items()}
    pairs = {s: PairData(sp, v, tables) for s, (sp, v) in splits.items()}
    return tables, votes, pairs


def input_dim(data, inputs):
    return data.features(torch.arange(1), inputs).shape[1]


# ---------------------------------------------------------------- Stage 1 experiments
# Each takes the speech and vote-record sizes and returns (model, loss_fn, optimizer).
# The model is an nn.ModuleDict with a "speech" and a "votes" head; its temperature is
# stored on the model.

def contrastive_heads(speech_dim, vote_dim, proj_dim=128, act=nn.ReLU, dropout=0.0,
                      temperature=0.07):
    def head(in_dim, hidden):
        layers = [nn.Linear(in_dim, hidden), act()]
        if dropout:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers, nn.Linear(hidden, proj_dim))
    model = nn.ModuleDict({"speech": head(speech_dim, 256), "votes": head(vote_dim, 64)})
    model.temperature = temperature
    return model


def contrastive_relu_adam(speech_dim, vote_dim, proj_dim=128):
    """Contrastive pretraining with 2-layer ReLU heads and Adam: the reference setup."""
    model = contrastive_heads(speech_dim, vote_dim, proj_dim)
    return model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


def contrastive_gelu_adam(speech_dim, vote_dim, proj_dim=128):
    """ReLU -> GELU: does the activation matter?"""
    model = contrastive_heads(speech_dim, vote_dim, proj_dim, act=nn.GELU)
    return model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


def contrastive_relu_sgd(speech_dim, vote_dim, proj_dim=128):
    """Adam -> SGD with momentum: does the optimizer matter?"""
    model = contrastive_heads(speech_dim, vote_dim, proj_dim)
    return model, nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.05,
                                                         momentum=0.9)


def contrastive_relu_adam_t005(speech_dim, vote_dim, proj_dim=128):
    """Temperature 0.07 -> 0.05: how sharp should the contrast be?"""
    model = contrastive_heads(speech_dim, vote_dim, proj_dim, temperature=0.05)
    return model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


def contrastive_relu_adam_dropout(speech_dim, vote_dim, proj_dim=128):
    """Dropout 0.2 in both heads: are we overfitting?"""
    model = contrastive_heads(speech_dim, vote_dim, proj_dim, dropout=0.2)
    return model, nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


STAGE1 = {
    "contrastive_relu_adam": contrastive_relu_adam,
    "contrastive_gelu_adam": contrastive_gelu_adam,
    "contrastive_relu_sgd": contrastive_relu_sgd,
    "contrastive_relu_adam_t0.05": contrastive_relu_adam_t005,
    "contrastive_relu_adam_dropout": contrastive_relu_adam_dropout,
}


# ---------------------------------------------------------------- Stage 2 experiments
# Each takes the input size and returns (model, loss_fn, optimizer). The EXPERIMENTS
# table says which features each one reads.

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


def contrastive_probe(in_dim):
    """Same head on the frozen Stage 1 speech projection + resolution."""
    model = prediction_head(in_dim)
    return model, nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


def contrastive_plus_history(in_dim):
    """Same head on Stage 1 speech projection + resolution + last-session vote."""
    model = prediction_head(in_dim)
    return model, nn.BCELoss(), torch.optim.Adam(model.parameters(), lr=1e-3)


EXPERIMENTS = {
    "probe_no_contrastive": (probe_no_contrastive, ["speech", "resolution"]),
    "history_mlp": (history_mlp, ["resolution", "history"]),
    "probe_plus_history": (probe_plus_history, ["speech", "resolution", "history"]),
    "contrastive_probe": (contrastive_probe, ["speech_proj", "resolution"]),
    "contrastive_plus_history": (contrastive_plus_history,
                                 ["speech_proj", "resolution", "history"]),
}


# ---------------------------------------------------------------- Stage 1 training

def clip_logits(model, s, v):
    """Cosine similarity of every speech to every vote record, over the temperature."""
    a = F.normalize(model["speech"](s), dim=1)
    b = F.normalize(model["votes"](v), dim=1)
    return a @ b.T / model.temperature


def clip_loss(loss_fn, logits):
    """Cross-entropy in both directions: speech -> record and record -> speech."""
    labels = torch.arange(len(logits), device=logits.device)
    return (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2


@torch.no_grad()
def retrieval(model, loss_fn, pairs, device):
    """Within-session retrieval of each speech's own vote record, plus mean loss."""
    model.eval()
    n = top1 = top10 = chance1 = chance10 = loss = 0.0
    batches = 0
    for idx in pairs.batches(max_batch=10_000):  # whole sessions, unshuffled
        logits = clip_logits(model, pairs.S[idx].to(device), pairs.V[idx].to(device))
        loss += clip_loss(loss_fn, logits).item()
        batches += 1
        rank = (logits > logits.diag()[:, None]).sum(dim=1)  # 0 = own record is closest
        k = len(idx)
        n += k
        top1 += (rank == 0).sum().item()
        top10 += (rank < 10).sum().item()
        chance1 += 1
        chance10 += min(10, k)
    return {"loss": loss / batches, "top1": top1 / n, "top10": top10 / n,
            "chance_top1": chance1 / n, "chance_top10": chance10 / n}


def train_contrastive(model, loss_fn, optimizer, pairs, epochs=300, patience=30,
                      max_batch=256, device=None, log=print, log_every=25, seed=SEED):
    """Stage 1 training loop. Returns (best model, per-epoch history DataFrame).

    Each epoch visits every training session once, in random order (one batch per
    session). Keeps the weights with the lowest val loss and stops after `patience`
    epochs without improvement.
    """
    device = device or pick_device()
    model.to(device)
    gen = torch.Generator().manual_seed(seed)
    history, best, best_state, stale = [], float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for idx in pairs["train"].batches(gen, max_batch):
            logits = clip_logits(model, pairs["train"].S[idx].to(device),
                                 pairs["train"].V[idx].to(device))
            loss = clip_loss(loss_fn, logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1

        val = retrieval(model, loss_fn, pairs["val"], device)
        history.append({"epoch": epoch, "train_loss": total / batches, "val_loss": val["loss"],
                        "val_top1": val["top1"], "val_top10": val["top10"]})
        if epoch == 1 or epoch % log_every == 0:
            log(f"  epoch {epoch:3d}  train loss {total / batches:.3f}  val loss {val['loss']:.3f}"
                f"  val top-1 {val['top1']:.3f}  top-10 {val['top10']:.3f}"
                f"  (chance {val['chance_top1']:.3f} / {val['chance_top10']:.3f})")

        if val["loss"] < best:
            best, stale = val["loss"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                log(f"  no val improvement for {patience} epochs, stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


@torch.no_grad()
def project_speeches(model, S, device):
    """Frozen Stage 1 speech head applied to every cached speech (L2-normalised)."""
    model.eval()
    return F.normalize(model["speech"](S.to(device)), dim=1).cpu()


# ---------------------------------------------------------------- Stage 2 training

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
          patience=3, device=None, log=print, seed=SEED):
    """Stage 2 training loop. Returns (best model, per-epoch history DataFrame).

    Each epoch: one pass over shuffled training votes, then val loss and val macro-F1
    (threshold tuned on val). Keeps the weights from the best val macro-F1 epoch and
    stops after `patience` epochs without improvement.
    """
    device = device or pick_device()
    model.to(device)
    train_data, val = data["train"], data["val"]
    gen = torch.Generator().manual_seed(seed)
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


def score_rows(name, data, decisions, threshold, seconds, seed):
    """Final Stage 2 scores (one row per split x subset), as in results/baselines.csv."""
    rows = []
    for split in EVAL_SPLITS:
        d = data[split]
        y = d.y.numpy().astype(np.int8)
        for subset, mask in d.subsets:
            rows.append({"baseline": name, "seed": seed, "split": split, "subset": subset,
                         "n": int(mask.sum()), **scores(y[mask], decisions[split][mask]),
                         "threshold": threshold, "seconds": round(seconds, 1)})
    return rows


def decide(probs, data):
    """Yes/No decisions per split, with the threshold tuned on val."""
    threshold = best_threshold(data["val"].y.numpy(), probs["val"])
    return {s: (p >= threshold).astype(np.int8) for s, p in probs.items()}, round(threshold, 4)


# ---------------------------------------------------------------- hybrid
# Uses the history model where a last-session vote exists and the speech model where it
# does not. No new training: it reuses the two trained models' P(Yes). The history
# model keeps its own val-tuned threshold; the speech model's threshold is chosen to
# maximise the hybrid's macro-F1 on all val votes (val only, so test stays untouched).
# Macro-F1 does not add up over subsets, so the cut-off that is best for the speech
# model on its own votes is not the best one for the combined predictions.

HYBRIDS = {"hybrid_history_or_speech": ("history_mlp", "contrastive_probe")}


def hybrid_decisions(data, p_history, t_history, p_speech):
    def combine(split, t_speech):
        no_history = dict(data[split].subsets)["no_history"]
        return np.where(no_history, p_speech[split] >= t_speech,
                        p_history[split] >= t_history).astype(np.int8)

    y_val = data["val"].y.numpy()
    nh_val = dict(data["val"].subsets)["no_history"]
    grid = np.unique(np.quantile(p_speech["val"][nh_val], np.linspace(0.01, 0.99, 99)))
    t_speech = float(max(grid, key=lambda t: scores(y_val, combine("val", t))["macro_f1"]))
    return {s: combine(s, t_speech) for s in EVAL_SPLITS}, f"{t_history:.4f}/{t_speech:.4f}"


# ---------------------------------------------------------------- run

def run_stage1(name, pairs, device, epochs, patience, seed):
    print(f"running {name}  (Stage 1, contrastive, seed {seed})")
    torch.manual_seed(seed)
    t0 = time.time()
    model, loss_fn, optimizer = STAGE1[name](pairs["train"].S.shape[1], pairs["train"].V.shape[1])
    model, hist = train_contrastive(model, loss_fn, optimizer, pairs, epochs, patience,
                                    device=device, seed=seed)
    rows = [{"experiment": name, "seed": seed, "split": s,
             **retrieval(model, loss_fn, pairs[s], device),
             "epochs": len(hist), "seconds": round(time.time() - t0, 1)} for s in EVAL_SPLITS]
    return model, hist.assign(seed=seed), rows


def run(names, epochs=20, batch_size=1024, patience=3, device=None,
        stage1="contrastive_relu_adam", stage1_epochs=300, stage1_patience=30, seeds=(SEED,)):
    """Run each experiment once per seed. Returns (Stage 2 rows, Stage 1 rows)."""
    device = pick_device(device)
    print(f"device: {device}")
    tables, data, pairs = load_data()
    RESULTS.mkdir(exist_ok=True)
    stage2_rows, stage1_rows, histories = [], [], {}

    def log_history(name, hist):
        histories.setdefault(name, []).append(hist)

    for seed in seeds:
        trained, decided = {}, {}

        def stage1_model(name):
            if name not in trained:
                trained[name], hist, rows = run_stage1(name, pairs, device, stage1_epochs,
                                                       stage1_patience, seed)
                log_history(name, hist)
                stage1_rows.extend(rows)
            return trained[name]

        def stage2_decisions(name):
            if name in decided:
                return decided[name]
            if name in HYBRIDS:
                history, speech = (stage2_decisions(n) for n in HYBRIDS[name])
                decisions, threshold = hybrid_decisions(data, history[3], history[1], speech[3])
                decided[name] = (decisions, threshold, 0.0, None)
            else:
                fn, inputs = EXPERIMENTS[name]
                if "speech_proj" in inputs:
                    P = project_speeches(stage1_model(stage1), tables[0], device)
                    for d in data.values():
                        d.P = P
                print(f"running {name}  (Stage 2, inputs: {', '.join(inputs)}"
                      + (f"; Stage 1 = {stage1}" if "speech_proj" in inputs else "")
                      + f", seed {seed})")
                torch.manual_seed(seed)
                t0 = time.time()
                model, loss_fn, optimizer = fn(input_dim(data["train"], inputs))
                model, hist = train(model, loss_fn, optimizer, data, inputs, epochs,
                                    batch_size, patience, device, seed=seed)
                log_history(name, hist.assign(seed=seed))
                probs = {s: predict(model, data[s], inputs, device) for s in EVAL_SPLITS}
                decisions, threshold = decide(probs, data)
                decided[name] = (decisions, threshold, time.time() - t0, probs)
            return decided[name]

        for name in names:
            if name in STAGE1:
                stage1_model(name)
            else:
                decisions, threshold, seconds, _ = stage2_decisions(name)
                stage2_rows += score_rows(name, data, decisions, threshold, seconds, seed)

    for name, hists in histories.items():
        pd.concat(hists, ignore_index=True).to_csv(RESULTS / f"{name}.csv", index=False)
    return pd.DataFrame(stage2_rows), pd.DataFrame(stage1_rows)


def save(results, path, key):
    """Write results, keeping earlier rows for experiments not re-run."""
    if path.exists() and len(results):
        old = pd.read_csv(path)
        results = pd.concat([old[~old[key].isin(results[key])], results], ignore_index=True)
    if len(results):
        results.to_csv(path, index=False)
    return results


def mean_std(df, index, columns, value):
    """Table of 'mean ± std' over seeds (just the mean when there is one seed)."""
    g = df.groupby([*index, *columns], sort=False)[value]
    mean, std, n = g.mean(), g.std(), g.size()
    cell = mean.map("{:.3f}".format)
    many = n > 1
    cell[many] = cell[many] + " ± " + std[many].map("{:.3f}".format)
    return cell.unstack(columns) if columns else cell


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", choices=[*STAGE1, *EXPERIMENTS, *HYBRIDS], action="append",
                   help="run only this experiment (repeatable); default: all")
    p.add_argument("--seeds", type=int, default=1,
                   help="run every experiment this many times with different seeds")
    p.add_argument("--epochs", type=int, default=20, help="Stage 2 max epochs")
    p.add_argument("--batch-size", type=int, default=1024, help="Stage 2 batch size")
    p.add_argument("--patience", type=int, default=3, help="Stage 2 early-stopping patience")
    p.add_argument("--stage1", choices=list(STAGE1), default="contrastive_relu_adam",
                   help="Stage 1 model whose speech head the contrastive_* Stage 2 runs use")
    p.add_argument("--stage1-epochs", type=int, default=300)
    p.add_argument("--stage1-patience", type=int, default=30)
    p.add_argument("--device", help="cpu, cuda or mps (default: auto)")
    a = p.parse_args()

    names = a.experiment or [*STAGE1, *EXPERIMENTS, *HYBRIDS]
    seeds = [SEED + i for i in range(a.seeds)]
    stage2, stage1 = run(names, a.epochs, a.batch_size, a.patience, a.device, a.stage1,
                         a.stage1_epochs, a.stage1_patience, seeds)
    over = f" (mean ± std over {a.seeds} seeds)" if a.seeds > 1 else ""

    if len(stage1):
        save(stage1, RESULTS / "retrieval.csv", "experiment")
        chance = stage1.groupby("split", sort=False)[["chance_top1", "chance_top10"]].first()
        print(f"\nStage 1 retrieval, within session{over}")
        for metric in ["top1", "top10"]:
            table = mean_std(stage1, ["experiment"], ["split"], metric)
            table.loc["chance"] = chance[f"chance_{metric}"].map("{:.3f}".format)
            print(f"\n{metric}:\n{table.to_string()}")
    if len(stage2):
        stage2 = save(stage2, RESULTS / "summary.csv", "baseline")
        print(f"\nStage 2 macro-F1, test split{over}")
        print(mean_std(stage2.query("split == 'test'"), ["baseline"], ["subset"],
                       "macro_f1").to_string())
        baselines = RESULTS / "baselines.csv"
        if baselines.exists():
            b = pd.read_csv(baselines).query("split == 'test'")
            print("\nbaselines, test macro-F1:")
            print(b.pivot_table(index="baseline", columns="subset", values="macro_f1",
                                sort=False).round(3).to_string())
    print(f"\nwritten to {RESULTS}")
