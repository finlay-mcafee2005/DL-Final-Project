"""Figures for the report, drawn from the CSVs in results/.

    uv run python plots.py

Writes PNG (for slides and previews) and PDF (vector, for report.tex) versions of each
figure to report/figures/:

    loss_curves      train vs validation loss per epoch: Stage 1 (contrastive) and the
                     two Stage 2 probes. One thin line per seed.
    retrieval        Stage 1 retrieval on test, each contrastive variant against chance
                     (mean ± std over seeds)
    results          test macro-F1 for every model and baseline, on all votes,
                     contested votes and votes with no history (mean ± std over seeds)
    by_issue         test macro-F1 on contested votes by issue area: history alone,
                     speech alone, and speech + history

A figure whose inputs are missing is skipped with a message, so the script can run at
any point in the project. Run comparison.py and un_contrastive.py first.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"

# Colours from a colour-blind-validated palette, used in fixed order.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#8a8985"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

LABELS = {
    "always_yes": "Always Yes",
    "tfidf_logreg": "TF-IDF + logistic regression",
    "random_forest": "Random forest (embeddings)",
    "last_session_vote": "Last session's vote",
    "probe_no_contrastive": "Speech probe (raw embeddings)",
    "contrastive_probe": "Speech probe (contrastive)",
    "history_mlp": "History MLP",
    "probe_plus_history": "Raw speech + history",
    "contrastive_plus_history": "Contrastive speech + history",
    "hybrid_history_or_speech": "Hybrid: history, else speech",
    "contrastive_relu_adam": "ReLU, Adam, τ 0.07 (reference)",
    "contrastive_gelu_adam": "GELU",
    "contrastive_relu_sgd": "SGD",
    "contrastive_relu_adam_t0.05": "Temperature 0.05",
    "contrastive_relu_adam_dropout": "Dropout 0.2",
    "middle_east": "Middle East", "nuclear": "Nuclear", "disarmament": "Disarmament",
    "human_rights": "Human rights", "colonialism": "Colonialism",
    "development": "Development", "other": "Other",
}
# Rows of the results figure, top to bottom: baselines, speech only, with history.
RESULT_ORDER = ["always_yes", "tfidf_logreg", "random_forest", "probe_no_contrastive",
                "contrastive_probe", "last_session_vote", "history_mlp",
                "probe_plus_history", "contrastive_plus_history", "hybrid_history_or_speech"]
BASELINES = {"always_yes", "tfidf_logreg", "random_forest", "last_session_vote"}
SUBSETS = [("all", "All votes"), ("contested", "Contested votes"),
           ("no_history", "No history (new resolutions)")]


# ---------------------------------------------------------------- helpers

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "legend.frameon": False})
    return plt


def _style(ax, grid_axis="both"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)


def _read(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def _save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"  {name}")


def _mean_std(df, keys, value):
    g = df.groupby(keys, sort=False)[value]
    out = g.agg(["mean", "std", "size"]).reset_index()
    out["std"] = out["std"].fillna(0)
    return out


# ---------------------------------------------------------------- figures

def loss_curves():
    """Train vs validation loss: Stage 1 reference and the two Stage 2 probes."""
    from matplotlib.ticker import MaxNLocator
    panels = [("contrastive_relu_adam", "Stage 1: contrastive (reference)", "contrastive loss"),
              ("probe_no_contrastive", "Stage 2: speech probe, raw embeddings", "binary cross-entropy"),
              ("contrastive_probe", "Stage 2: speech probe, contrastive", "binary cross-entropy")]
    hists = [(_read(f"{name}.csv"), title, ylabel) for name, title, ylabel in panels]
    if all(h is None for h, _, _ in hists):
        return print("  loss_curves: skipped (no training histories)")
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, (hist, title, ylabel) in zip(axes, hists):
        ax.set_title(title, loc="left", color=INK)
        _style(ax)
        if hist is None:
            ax.text(0.5, 0.5, "not run yet", ha="center", transform=ax.transAxes, color=INK2)
            continue
        if "seed" not in hist:
            hist = hist.assign(seed=0)
        for _, h in hist.groupby("seed"):
            ax.plot(h["epoch"], h["train_loss"], color=BLUE, linewidth=1.2, alpha=0.7)
            ax.plot(h["epoch"], h["val_loss"], color=ORANGE, linewidth=1.2, alpha=0.7)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].plot([], [], color=BLUE, linewidth=2, label="train")
    axes[0].plot([], [], color=ORANGE, linewidth=2, label="validation")
    fig.legend(loc="upper left", bbox_to_anchor=(0.0, 1.08), ncol=2)
    n_seeds = max(h["seed"].nunique() for h, _, _ in hists if h is not None and "seed" in h)
    fig.text(0.0, -0.04, f"One line per seed ({n_seeds} seeds). Validation loss rising while "
             "training loss falls marks overfitting.", color=INK2, fontsize=7)
    fig.tight_layout()
    _save(fig, "loss_curves")
    plt.close(fig)


def retrieval():
    """Stage 1: within-session retrieval on test for each variant, against chance."""
    r = _read("retrieval.csv")
    if r is None:
        return print("  retrieval: skipped (no results/retrieval.csv)")
    r = r[r["split"] == "test"]
    if "seed" not in r:
        r = r.assign(seed=0)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 2.4), sharey=True)
    for ax, metric, title in zip(axes, ["top1", "top10"], ["Top-1", "Top-10"]):
        m = _mean_std(r, ["experiment"], metric)
        m = m.iloc[::-1].reset_index(drop=True)  # reference model on top
        y = np.arange(len(m))
        ax.errorbar(m["mean"], y, xerr=m["std"], fmt="o", color=BLUE, ms=6,
                    elinewidth=1.5, capsize=0)
        chance = r[f"chance_{metric}"].iloc[0]
        ax.axvline(chance, color=GRAY, linestyle="--", linewidth=1)
        ax.text(chance, -0.75, " chance", color=INK2, fontsize=7, va="center")
        ax.set_ylim(-1.0, len(m) - 0.5)
        ax.set_yticks(y, [LABELS.get(e, e) for e in m["experiment"]])
        ax.set_xlim(0, max(m["mean"].max() + m["std"].max(), chance) * 1.15)
        ax.set_title(f"{title}: own vote record found", loc="left", color=INK)
        ax.set_xlabel("share of test speeches")
        _style(ax, "x")
    n = r["seed"].nunique()
    fig.text(0.0, -0.06, f"Test sessions 2016-2025, within-session retrieval. Mean ± std over "
             f"{n} seed{'s' if n > 1 else ''}.", color=INK2, fontsize=7)
    fig.tight_layout()
    _save(fig, "retrieval")
    plt.close(fig)


def results():
    """Test macro-F1 for every model, on all, contested and no-history votes."""
    frames = [f for f in [_read("baselines.csv"), _read("summary.csv")] if f is not None]
    if not frames:
        return print("  results: skipped (no results/baselines.csv or summary.csv)")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["split"] == "test"]
    order = [b for b in RESULT_ORDER if b in set(df["baseline"])][::-1]
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.4), sharey=True, sharex=True)
    shown = df[df["baseline"].isin(order)]["macro_f1"]
    axes[0].set_xlim(np.floor(shown.min() * 20) / 20 - 0.02, np.ceil(shown.max() * 20) / 20 + 0.02)
    for ax, (subset, title) in zip(axes, SUBSETS):
        m = _mean_std(df[df["subset"] == subset], ["baseline"], "macro_f1").set_index("baseline")
        for i, name in enumerate(order):
            if name not in m.index:
                continue
            color = GRAY if name in BASELINES else BLUE
            ax.errorbar(m.loc[name, "mean"], i, xerr=m.loc[name, "std"], fmt="o",
                        color=color, ms=6, elinewidth=1.5, capsize=0)
            ax.text(m.loc[name, "mean"], i + 0.28, f"{m.loc[name, 'mean']:.2f}",
                    ha="center", fontsize=6.5, color=INK2)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("macro-F1")
        _style(ax, "x")
    axes[0].set_yticks(np.arange(len(order)), [LABELS.get(b, b) for b in order])
    axes[0].errorbar([], [], fmt="o", color=GRAY, label="baseline")
    axes[0].errorbar([], [], fmt="o", color=BLUE, label="neural model (mean ± std)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.0, -0.08), ncol=2, fontsize=7)
    fig.text(0.42, -0.06, "Test sessions 2016-2025. All panels share one scale.",
             color=INK2, fontsize=7)
    fig.tight_layout()
    _save(fig, "results")
    plt.close(fig)


def by_issue():
    """Contested test votes by issue: history alone, speech alone, speech + history."""
    df = _read("by_issue.csv")
    if df is None:
        return print("  by_issue: skipped (no results/by_issue.csv)")
    models = [("last_session_vote", BLUE, "o"), ("contrastive_probe", ORANGE, "s"),
              ("contrastive_plus_history", AQUA, "^")]
    df = df[(df["subset"] == "contested") & df["baseline"].isin([m for m, _, _ in models])]
    if df.empty:
        return print("  by_issue: skipped (none of the three models in results/by_issue.csv)")
    m = _mean_std(df, ["baseline", "issue"], "macro_f1")
    n = df.drop_duplicates("issue").set_index("issue")["n"]
    ref = m[m["baseline"] == "last_session_vote"].set_index("issue")["mean"]
    issues = list(ref.sort_values().index) if len(ref) else list(n.index)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, (name, color, marker) in enumerate(models):
        d = m[m["baseline"] == name].set_index("issue").reindex(issues)
        y = np.arange(len(issues)) + (k - 1) * 0.22
        ax.errorbar(d["mean"], y, xerr=d["std"], fmt=marker, color=color, ms=6,
                    elinewidth=1.5, capsize=0, label=LABELS[name])
    ax.set_yticks(np.arange(len(issues)),
                  [f"{LABELS.get(i, i)}  (n = {n.get(i, 0):,})" for i in issues])
    ax.set_xlabel("macro-F1, contested test votes")
    ax.set_title("Accuracy by issue area", loc="left", color=INK)
    _style(ax, "x")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=7)
    fig.tight_layout()
    _save(fig, "by_issue")
    plt.close(fig)


FIGURES_ALL = [loss_curves, retrieval, results, by_issue]

if __name__ == "__main__":
    print(f"writing figures to {FIGURES}")
    for make in FIGURES_ALL:
        make()
