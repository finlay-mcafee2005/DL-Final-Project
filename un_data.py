"""Load and join both datasets, build time splits, flag contested resolutions.

Reads the raw data in data/ and writes a model-ready dataset to data/processed/:

    speeches.csv     one row per country-session speech that has votes
                     (speech_id, iso3, session, year, speaker, text, split, heldout_country)
    votes.csv        one row per (speech_id, undl_id): vote in {Y, N, A}, plus the
                     country's vote on the same issue in the last session (history)
    resolutions.csv  one row per resolution (title, agenda, subjects, tallies, contested)

Alignment: the General Debate speech of session t (September of year t) is paired
with every recorded vote the country cast in session t. Only speeches with at least
one Y/N/A vote are kept.

Splits (by time, so the model never sees the future):
    train       sessions before 2010
    val         2010-2015
    test        2016 onward
A second test set holds out a few countries entirely: their speeches are flagged
`heldout_country` and must not be used for training or validation. They are
evaluated in the test period (`load_split("test_heldout")`).

Usage:
    uv run python un_data.py
    from un_data import load_split
    speeches, votes = load_split("train")   # "train" | "val" | "test" | "test_heldout"
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
OUT = DATA / "processed"
SEED = 42

VAL_START, TEST_START = 2010, 2016   # README split years
N_HELDOUT_COUNTRIES = 10
# A resolution is contested when at least this share of the votes cast were No or
# Abstain, i.e. there is a meaningful No/Abstain bloc.
CONTESTED_MIN_SHARE = 0.10

# The vote data codes some historical states differently from the speech files.
# Map vote codes onto the speech-file code for the same delegation.
VOTE_CODE_TO_SPEECH_CODE = {
    "GER": "DEU",  # Federal Republic of Germany (1973-1990)
    "SUN": "RUS",  # USSR (speeches are filed under RUS)
    "SCG": "YUG",  # Serbia and Montenegro (speeches filed under YUG)
    "EAT": "TZA",  # Tanganyika, before union with Zanzibar in 1964
}
VALID_VOTES = ["Y", "N", "A"]  # X = non-voting, which is not a vote


# ---------------------------------------------------------------- loading

def load_speeches():
    rows = []
    for path in sorted((DATA / "TXT").glob("Session */*.txt")):
        m = re.fullmatch(r"([A-Z_]+)_(\d+)_(\d{4})\.txt", path.name)
        if not m:  # skips macOS ._ resource-fork files
            continue
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        rows.append({"iso3": m[1], "session": int(m[2]), "year": int(m[3]), "text": text})
    return pd.DataFrame(rows)


def load_votes():
    # Tracked in git as .csv.gz (the raw CSV is over GitHub's 100 MB file limit).
    v = pd.read_csv(DATA / "2026_10_15_votes_ga.csv.gz", dtype=str)
    # A few rows are missing the session; recover it from the resolution symbol.
    from_symbol = v["resolution"].str.extract(r"^A/RES/(\d+)/", expand=False)
    v["session"] = v["session"].fillna(from_symbol)
    # Keep regular sessions only (drops special "sp" and emergency "emsp" sessions).
    v = v[v["session"].fillna("").str.fullmatch(r"\d+")].copy()
    v["session"] = v["session"].astype(int)
    v["undl_id"] = v["undl_id"].astype(int)
    v["iso3"] = v["ms_code"].replace(VOTE_CODE_TO_SPEECH_CODE)
    return v


def load_speakers():
    x = pd.read_excel(DATA / "Speakers_by_session.xlsx")
    x = x.rename(columns={"Session": "session", "ISO Code": "iso3", "Country": "country",
                          "Name of Person Speaking": "speaker", "Post": "speaker_post"})
    x = x[["session", "iso3", "country", "speaker", "speaker_post"]]
    return x.drop_duplicates(["session", "iso3"])


# ---------------------------------------------------------------- resolutions

def _norm(s):
    return s.fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()


def build_resolutions(votes_raw):
    r = votes_raw.drop_duplicates("undl_id")[
        ["undl_id", "session", "date", "resolution", "draft", "meeting", "title",
         "agenda_title", "subjects", "total_yes", "total_no", "total_abstentions",
         "total_non_voting", "total_ms", "undl_link"]].copy()
    for c in ["total_yes", "total_no", "total_abstentions", "total_non_voting", "total_ms"]:
        r[c] = pd.to_numeric(r[c]).astype("Int64")
    cast = (r["total_yes"] + r["total_no"] + r["total_abstentions"]).astype(float)
    r["yes_share"] = (r["total_yes"] / cast.replace(0, np.nan)).round(4)
    r["contested"] = (1 - r["yes_share"]) >= CONTESTED_MIN_SHARE
    return r.sort_values(["session", "date", "undl_id"]).reset_index(drop=True)


def match_previous(resolutions):
    """Link each resolution to its counterpart(s) in the most recent earlier session.

    Resolutions get new symbols every year, so recurring items are matched on the
    title (e.g. "Question of Palestine"), falling back to the agenda item. Returns
    (undl_id, prev_undl_id, prev_session, match) pairs.
    """
    r = resolutions[["undl_id", "session"]].copy()
    r["title_key"] = _norm(resolutions["title"].str.split(" : ").str[0])
    r["agenda_key"] = _norm(resolutions["agenda_title"])

    def link(key):
        k = r[r[key] != ""][["undl_id", "session", key]]
        pairs = k.merge(k, on=key, suffixes=("", "_prev"))
        pairs = pairs[pairs["session_prev"] < pairs["session"]]
        latest = pairs.groupby("undl_id")["session_prev"].transform("max")
        pairs = pairs[pairs["session_prev"] == latest]
        return pairs[["undl_id", "undl_id_prev", "session_prev"]]

    by_title = link("title_key").assign(match="title")
    by_agenda = link("agenda_key").assign(match="agenda")
    by_agenda = by_agenda[~by_agenda["undl_id"].isin(by_title["undl_id"])]
    out = pd.concat([by_title, by_agenda], ignore_index=True)
    return out.rename(columns={"undl_id_prev": "prev_undl_id", "session_prev": "prev_session"})


def add_last_session_vote(votes, votes_all, links):
    """Add the country's vote on the matched resolution(s) in the previous session.

    History comes from all recorded votes (even sessions where the country gave no
    speech). If several previous resolutions match, the country's most common vote
    among them is used. Missing when there is no match or the country did not vote.
    """
    hist = votes_all.rename(columns={"undl_id": "prev_undl_id", "vote": "prev_vote"})
    h = links.merge(hist[["iso3", "prev_undl_id", "prev_vote"]], on="prev_undl_id")
    h = (h.groupby(["iso3", "undl_id", "prev_session", "match", "prev_vote"]).size()
          .rename("n").reset_index()
          .sort_values(["iso3", "undl_id", "n", "prev_vote"], ascending=[True, True, False, False])
          .drop_duplicates(["iso3", "undl_id"]))
    h = h.rename(columns={"prev_vote": "last_vote", "prev_session": "last_vote_session",
                          "match": "last_vote_match"}).drop(columns="n")
    out = votes.merge(h, on=["iso3", "undl_id"], how="left")
    out["last_vote_session"] = out["last_vote_session"].astype("Int64")
    return out


# ---------------------------------------------------------------- splits

def time_split(year):
    return np.where(year < VAL_START, "train", np.where(year < TEST_START, "val", "test"))


def pick_heldout_countries(speeches):
    """Random countries that have speeches in all three time periods."""
    periods = speeches.groupby("iso3")["split"].nunique()
    eligible = sorted(periods[periods == 3].index)
    rng = np.random.default_rng(SEED)
    return sorted(rng.choice(eligible, N_HELDOUT_COUNTRIES, replace=False))


# ---------------------------------------------------------------- build

def build():
    speeches = load_speeches()
    votes_raw = load_votes()
    resolutions = build_resolutions(votes_raw)

    votes_all = votes_raw[votes_raw["ms_vote"].isin(VALID_VOTES)]
    votes_all = (votes_all[["iso3", "session", "undl_id", "ms_vote"]]
                 .rename(columns={"ms_vote": "vote"})
                 # merged codes (e.g. SUN and RUS in 1991) could overlap on a resolution
                 .drop_duplicates(["iso3", "undl_id"]))

    # Keep only speeches with at least one real vote in the same session.
    n_votes = votes_all.groupby(["iso3", "session"]).size().rename("n_votes").reset_index()
    speeches = speeches.merge(n_votes, on=["iso3", "session"], how="inner")
    speeches = speeches.merge(load_speakers(), on=["iso3", "session"], how="left")
    speeches["speech_id"] = speeches["iso3"] + "_" + speeches["session"].astype(str)
    speeches["n_words"] = speeches["text"].str.split().str.len()
    speeches["split"] = time_split(speeches["year"])
    heldout = pick_heldout_countries(speeches)
    speeches["heldout_country"] = speeches["iso3"].isin(heldout)
    speeches = speeches.sort_values(["session", "iso3"])[
        ["speech_id", "iso3", "country", "session", "year", "speaker", "speaker_post",
         "n_words", "n_votes", "split", "heldout_country", "text"]]

    votes = votes_all.merge(speeches[["speech_id", "iso3", "session"]],
                            on=["iso3", "session"], how="inner")
    votes = add_last_session_vote(votes, votes_all, match_previous(resolutions))
    votes = votes[["speech_id", "undl_id", "vote", "last_vote", "last_vote_session",
                   "last_vote_match"]].sort_values(["speech_id", "undl_id"])
    resolutions = resolutions[resolutions["undl_id"].isin(votes["undl_id"])]

    OUT.mkdir(parents=True, exist_ok=True)
    speeches.to_csv(OUT / "speeches.csv", index=False)
    votes.to_csv(OUT / "votes.csv", index=False)
    resolutions.to_csv(OUT / "resolutions.csv", index=False)
    return speeches, votes, resolutions


def load_processed():
    speeches = pd.read_csv(OUT / "speeches.csv", keep_default_na=False, na_values=[""])
    votes = pd.read_csv(OUT / "votes.csv", dtype={"last_vote_session": "Int64"})
    resolutions = pd.read_csv(OUT / "resolutions.csv")
    return speeches, votes, resolutions


def load_split(split):
    """Return (speeches, votes) for "train", "val", "test" or "test_heldout".

    train/val/test exclude the held-out countries; "test_heldout" is the held-out
    countries in the test period. Votes come joined with resolution info.
    """
    speeches, votes, resolutions = load_processed()
    if split == "test_heldout":
        keep = (speeches["split"] == "test") & speeches["heldout_country"]
    elif split in ("train", "val", "test"):
        keep = (speeches["split"] == split) & ~speeches["heldout_country"]
    else:
        raise ValueError(f"unknown split {split!r}")
    speeches = speeches[keep].reset_index(drop=True)
    votes = votes[votes["speech_id"].isin(speeches["speech_id"])]
    votes = votes.merge(resolutions, on="undl_id", how="left").reset_index(drop=True)
    return speeches, votes


if __name__ == "__main__":
    speeches, votes, resolutions = build()
    print(f"speeches: {len(speeches):,}  votes: {len(votes):,}  resolutions: {len(resolutions):,}"
          f"  ({resolutions.contested.mean():.0%} contested)")
    print(f"held-out countries: {', '.join(sorted(speeches[speeches.heldout_country].iso3.unique()))}")
    print(f"last-session vote available for {votes.last_vote.notna().mean():.0%} of votes")
    for s in ["train", "val", "test", "test_heldout"]:
        sp, vo = load_split(s)
        yrs = f"{sp.year.min()}-{sp.year.max()}"
        mix = vo.vote.value_counts(normalize=True).round(3).to_dict()
        print(f"  {s:12s} {yrs}: {len(sp):6,} speeches, {len(vo):8,} votes, "
              f"{vo.contested.mean():.0%} contested, mix {mix}")
    print(f"written to {OUT}")
