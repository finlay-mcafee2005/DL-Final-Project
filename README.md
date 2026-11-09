# Predicting UN Votes from General Debate Speeches

**COMP 395 – Deep Learning · Final Project · Occidental College**

Every September, each UN member state gives a speech at the General Debate laying out what it cares about. A few months later, it casts dozens of recorded votes in the General Assembly. This project tests whether **what a country says predicts how it votes**, using a contrastive model that places speeches and votes in one shared embedding space.

> **Main question:** Can a country's General Debate speech in year *t* predict its General Assembly votes in session *t*, and does it add signal beyond the country's own voting history?

---

## Why this is interesting

- **Speeches are cheap talk; votes are commitments.** How closely the two line up is a real political-science question, usually studied with hand-coded features or bag-of-words rather than learned representations.
- **It's a representation-learning problem.** A long diplomatic speech has to become a vector that understands issues (Middle East, disarmament, human rights, colonialism, development) well enough to match it against specific resolutions.
- **Contrastive learning fits naturally.** Speeches and votes are two "views" of the same thing, a country's position. A CLIP-style objective pulls matched pairs together and pushes mismatched pairs apart.

Secondary questions, if time allows: which issue areas are speeches most and least predictive of, and which countries "say one thing and vote another"?

---

## Data

Both datasets are public, already cleaned, and keyed by country and year.

| Dataset | What it contains | Unit we use |
|---|---|---|
| [UN General Debate Corpus](https://doi.org/10.7910/DVN/0TJX8Y) (Baturo, Dasandi & Mikhaylov) | English text of every General Debate speech since 1970 | One speech per country per year |
| [UN General Assembly Voting Data](https://doi.org/10.7910/DVN/LEJUQZ) (Voeten) | Every recorded roll-call vote (Yes / No / Abstain) with resolution descriptions and issue codes | One (country, resolution, vote) triple |

**Alignment.** The speech from the September General Debate of year *t* is paired with every recorded vote in that same session.

**Splits.** Split by time so the model never sees the future:

| Split | Sessions |
|---|---|
| Train | before 2010 |
| Validation | 2010–2015 |
| Test | 2016 onward |

A second test set holds out a few countries entirely, to check the model isn't just memorizing country identity.

**Contested votes.** Roughly 80% of all GA votes are Yes, so results are also reported on *contested* resolutions (those with a meaningful No/Abstain bloc), where the real signal is.

---

## Method

The training setup mirrors Lab 3: every experiment is a function that returns `(model, loss_fn, optimizer)`, models are built from `nn.Sequential`, and one shared `train()` loop runs them all.

### Frozen text encoder, small trainable heads

A pretrained sentence-transformer embeds each speech (chunked, then mean-pooled) and each resolution description **once**, and the embeddings are cached to disk. Every experiment trains only small MLP heads on top, so runs take minutes on a laptop or Colab.

### Stage 1: contrastive pretraining (CLIP-style)

Two `nn.Sequential` projection heads map **speeches** and **vote records** (a country's per-issue voting profile for the session) into the same space. In a batch of *B* country-sessions, the *B × B* similarity matrix should peak on the diagonal: each speech matches its own country's votes. The loss is `nn.CrossEntropyLoss` over that matrix in both directions.

### Stage 2: vote prediction

The speech head is frozen. A new `nn.Sequential` takes `[speech embedding, resolution embedding]` and outputs P(Yes) through a final sigmoid, trained with `nn.BCELoss`, the same setup as Lab 3.

```python
def contrastive_relu_adam(speech_dim, vote_dim, proj_dim=128):
    """Contrastive pretraining with 2-layer ReLU heads and Adam: the reference setup."""
    model = nn.ModuleDict({
        "speech": nn.Sequential(nn.Linear(speech_dim, 256), nn.ReLU(), nn.Linear(256, proj_dim)),
        "votes":  nn.Sequential(nn.Linear(vote_dim, 64), nn.ReLU(), nn.Linear(64, proj_dim)),
    })
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, loss_fn, optimizer
```

---

## Experiments

Each experiment changes **one thing** from another.

| Experiment | What changes | Question it tests |
|---|---|---|
| `probe_no_contrastive` | Stage 2 only, on raw frozen embeddings | Does contrastive training add anything? |
| `contrastive_relu_adam` | Reference setup | Baseline contrastive model |
| `contrastive_gelu_adam` | ReLU → GELU | Does the activation matter? |
| `contrastive_relu_sgd` | Adam → SGD with momentum | Does the optimizer matter? |
| `contrastive_relu_adam_t0.05` | Temperature 0.07 → 0.05 | How sharp should the contrast be? |
| `contrastive_relu_adam_dropout` | Dropout in the heads | Are we overfitting? |
| `contrastive_plus_history` | Adds last-session votes to the Stage 2 input | Do speeches add signal beyond history? |

### Baselines

The bar to clear is "last year's vote," not random guessing.

| Baseline | Uses speech? | Uses history? | Purpose |
|---|---|---|---|
| `always_yes` | No | No | Shows how misleading raw accuracy is |
| `last_session_vote` | No | Yes | The strong, hard-to-beat baseline |
| `tfidf_logreg` | Yes | No | Classic NLP: is deep learning needed at all? |
| `random_forest` | Yes | No | Classical model on the same cached embeddings |

The headline comparison is **`contrastive_plus_history` vs. `last_session_vote`** on contested votes.

### Metrics

- **Vote prediction:** macro-F1 and balanced accuracy, on all votes and on contested votes
- **Retrieval:** top-1 and top-10 accuracy for matching a speech to the right country's voting record
- **Interpretability:** accuracy broken down by issue area, and correlation of learned country positions with established UN ideal-point estimates

---

## Repository layout

> Files below are planned and will be added as the project progresses.

| File | Purpose |
|---|---|
| `un_data.py` | Load and join both datasets, build time splits, flag contested resolutions |
| `embed_cache.py` | Compute and save speech and resolution embeddings |
| `un_contrastive.py` | Experiment functions + shared `train()` loop and logging |
| `comparison.py` | Classical baselines |
| `plots.py` | Overlaid loss and F1 curves for the report |
| `test_un.py` | Shape checks, no NaNs, no time leakage, beats `last_session_vote` |
| `results/` | Per-run training history (CSV) and a summary table |
| `report/` | Final report (`report.tex`, `report.pdf`, figures) |

---

## Getting started

```bash
git clone https://github.com/finlay-mcafee2005/DL-Final-Project.git
cd DL-Final-Project
pip install -r requirements.txt

python un_data.py                    # download, join, and split the data
python embed_cache.py                # cache sentence embeddings (run once)
python comparison.py                 # run the baselines
python un_contrastive.py             # run every experiment
python un_contrastive.py --experiment contrastive_relu_adam   # run just one
python -m pytest test_un.py -v       # run the tests
```

Logging goes to CSV in `results/` by default. Weights & Biases is optional (`--wandb`).

---

## Team and roles

Three build roles, with evaluation and the report shared by the whole group.

| Role | Owns | Focus |
|---|---|---|
| **A: Data & baselines** | `un_data.py`, `embed_cache.py`, `comparison.py`, leakage test | Correctness and no time leakage |
| **B: Stage 1, contrastive** | Speech and vote-record heads, contrastive experiments, retrieval metrics | A contrastive model that clearly beats random at matching speeches to countries |
| **C: Stage 2, prediction** | Prediction head, probe and history experiments, shared training/logging code | The headline comparison against `last_session_vote` |
| **Everyone** | Evaluation, analysis, plots, report, presentation | A clear answer to the main question |

---

## Timeline (4 weeks)

| Week | A | B | C | Together |
|---|---|---|---|---|
| 1 | Join data, splits, cache embeddings | Stage 1 loop on fake data | Stage 2 loop + logging on fake data | Agree on data format and metric names |
| 2 | Baselines, leakage test | Stage 1 on real data | Probe on real data | Check first results against baselines |
| 3 | Help run experiments | Contrastive experiments | History experiment | Issue- and country-level analysis |
| 4 | Data section | Stage 1 section | Stage 2 section | Results, plots, report, slides |

**Milestones**
- **End of Week 1:** data joined and split, embeddings cached
- **End of Week 2:** one contrastive model trained and compared against `last_session_vote`
- **End of Week 3:** all experiments done
- **End of Week 4:** report and presentation finished

---

## References

- Baturo, A., Dasandi, N., & Mikhaylov, S. (2017). Understanding state preferences with text as data: Introducing the UN General Debate corpus. *Research & Politics*.
- Voeten, E., Strezhnev, A., & Bailey, M. United Nations General Assembly Voting Data. Harvard Dataverse.
- Bailey, M., Strezhnev, A., & Voeten, E. (2017). Estimating dynamic state preferences from United Nations voting data. *Journal of Conflict Resolution*.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision (CLIP). *ICML*.
