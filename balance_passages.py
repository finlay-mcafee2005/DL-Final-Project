"""Balanced sample of audio passages: equal numbers per issue, spread evenly across countries.

Run after audio_tone.py, which writes results/audio_all/passages.csv (one row per passage,
Assembly President passages already removed, with z-scored measures and issue_<area> tags).

How the sample is drawn
  1. Each passage gets ONE stratum. Passages tagged with several issues go to the rarest
     of their issues (so the scarce issues keep every passage they can get); passages with
     no issue tag go to "other" (drop them with --no-other).
  2. The target (--n, default 5,000) is split equally across strata. A stratum with fewer
     passages than its share gives all it has, and the rest is shared equally among the
     others ("water-filling"), so strata are as equal as the data allow.
  3. Within each stratum the same rule spreads the passages across countries: every
     country gets an equal share, capped at what it has, so no country dominates an issue.
  4. Passages are drawn at random within each country and stratum (--seed, default 0).

Outputs (in the same folder as the input):
  balanced_passages.csv     the sample, with a `stratum` column
  balanced_counts.csv       passages per stratum and per country (available vs sampled)
  balanced_correlations.csv text-vs-voice correlations on the sample (audio_tone.correlations)
  balanced_issues.csv       tone by issue on the sample (audio_tone.by_issue)

Usage:
    uv run python balance_passages.py                    # 5,000 passages, issues + "other"
    uv run python balance_passages.py --n 3000 --no-other
    uv run python balance_passages.py --input results/audio/passages.csv   # English-only study
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from un_data import ISSUES

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "results" / "audio_all" / "passages.csv"
OTHER = "other"


def assign_strata(passages, include_other=True):
    """One stratum per passage: its rarest issue tag, else "other" (or NaN if excluded)."""
    tags = passages[[f"issue_{k}" for k in ISSUES]].astype(bool)
    tags.columns = list(ISSUES)
    rarest_first = tags.sum().sort_values(kind="stable").index.tolist()
    stratum = pd.Series(OTHER if include_other else np.nan, index=passages.index, dtype=object)
    for issue in reversed(rarest_first):  # rarest last, so it wins for multi-tagged passages
        stratum[tags[issue]] = issue
    return stratum


def waterfill(available, total):
    """Split `total` as equally as possible over the keys of `available`, never giving a
    key more than it has. Returns {key: count}; sums to min(total, sum(available))."""
    total = int(min(total, sum(available.values())))
    alloc = {k: 0 for k in available}
    remaining, keys = total, sorted(available, key=lambda k: (available[k], str(k)))
    for i, k in enumerate(keys):
        share = remaining // (len(keys) - i)
        alloc[k] = min(available[k], share)
        remaining -= alloc[k]
    for k in sorted(keys, key=lambda k: (-(available[k] - alloc[k]), str(k))):  # leftovers
        if remaining == 0:
            break
        if alloc[k] < available[k]:
            alloc[k] += 1
            remaining -= 1
    return alloc


def balanced_sample(passages, n=5000, include_other=True, seed=0):
    """The balanced sample (see the module docstring), with a `stratum` column."""
    rng = np.random.default_rng(seed)
    d = passages.assign(stratum=assign_strata(passages, include_other)).dropna(subset=["stratum"])
    per_stratum = waterfill(d["stratum"].value_counts().to_dict(), n)
    picks = []
    for stratum, quota in sorted(per_stratum.items()):
        s = d[d["stratum"] == stratum]
        per_country = waterfill(s["speech_id"].value_counts().to_dict(), quota)
        for speech_id, k in sorted(per_country.items()):
            rows = s.index[s["speech_id"] == speech_id].to_numpy()
            picks.extend(rng.choice(rows, size=k, replace=False))
    return d.loc[sorted(picks)].reset_index(drop=True)


def counts(passages, sample):
    """Available vs sampled passages per stratum and per country."""
    avail = passages.assign(stratum=assign_strata(passages, include_other=True))
    rows = []
    for level, col in [("stratum", "stratum"), ("country", "speech_id")]:
        a, s = avail[col].value_counts(), sample[col].value_counts()
        rows.append(pd.DataFrame({"level": level, "group": a.index, "available": a.to_numpy(),
                                  "sampled": s.reindex(a.index).fillna(0).astype(int).to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def main(input_path=DEFAULT_INPUT, n=5000, include_other=True, seed=0):
    import audio_tone
    input_path = Path(input_path)
    passages = pd.read_csv(input_path)
    sample = balanced_sample(passages, n, include_other, seed)
    out = input_path.parent
    c = counts(passages, sample)
    sample.to_csv(out / "balanced_passages.csv", index=False)
    c.to_csv(out / "balanced_counts.csv", index=False)
    audio_tone.correlations(sample).to_csv(out / "balanced_correlations.csv", index=False)
    audio_tone.by_issue(sample).to_csv(out / "balanced_issues.csv", index=False)

    strata = c[c["level"] == "stratum"].set_index("group")
    per_country = sample["speech_id"].value_counts()
    print(f"{len(sample):,} of {len(passages):,} passages from {sample['speech_id'].nunique()} speeches")
    print("\npassages per stratum (sampled / available):")
    for g, r in strata.sort_values("sampled").iterrows():
        print(f"  {g:<14} {r.sampled:>5} / {r.available}")
    print(f"\npassages per country: min {per_country.min()}, median {per_country.median():.0f},"
          f" max {per_country.max()}")
    if len(sample) < n:
        print(f"\nonly {len(sample):,} passages available{' outside other' if not include_other else ''};"
              " the sample is smaller than --n")
    short = strata[(strata["sampled"] < strata["sampled"].max()) & (strata["sampled"] == strata["available"])]
    if len(short):
        print("strata with fewer passages than an equal share (all used): " + ", ".join(short.index))
    print(f"\nwritten to {out}")
    return sample


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", default=DEFAULT_INPUT, help="passages.csv written by audio_tone.py")
    p.add_argument("--n", type=int, default=5000, help="number of passages (default 5,000)")
    p.add_argument("--no-other", action="store_true", help="leave out passages with no issue tag")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.input, a.n, not a.no_other, a.seed)
