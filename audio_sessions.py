"""Tone of voice vs votes across sessions 70-80 (2015-2025): one row per speech.

audio_tone.py studies the 2025 General Debate passage by passage (about 4 minutes per
speech). This script trades the passage-level detail for coverage: it measures each
speech as a whole (about 1 minute per speech), so every speech with floor audio from
2015 to 2025, about 2,100, can be compared with that country's votes in the same session.

Audio: the UN's floor recordings (the speaker's own voice, never an interpreter), at
    https://s3.amazonaws.com/downloads.unmultimedia.org/radio/library/ltd/mp3/ga/<year>/<session>_<ISO2>_FL.mp3
They exist from session 70 (2015); earlier years return 403.

Per speech (results cached in results/audio_sessions/speeches/<speech_id>.json):
  1. download the floor audio, drop the first 30 s and last 20 s (the Assembly
     President's introduction and thanks)
  2. voice: arousal, dominance, valence (audeering wav2vec2), averaged over 24 evenly
     spaced 10-second windows
  3. delivery: pitch level and variation, loudness, share of pauses, over the whole speech
  4. language and speaking rate: Whisper detects the language and transcribes 6 evenly
     spaced 30-second clips; words per second (characters for Chinese/Japanese)
  5. text sentiment: the English RoBERTa model on the official English text of the speech
     (data/processed/speeches.csv), the same text for every year, so sentiment is
     measured the same way whatever language the speech was given in

Analysis (results/audio_sessions/):
  tone.csv           one row per speech: measures, language, dissent on contested votes
  pooled.csv         Spearman of each measure with dissent over all country-sessions, after
                     standardising both within each session; 95% CI from a bootstrap that
                     resamples countries (a country appears once per year, so rows are not
                     independent)
  by_session.csv     the same correlation within each session
  country_level.csv  each country's average over the years: one row per country
  within_country.csv each country's deviations from its own average: does a country that
                     speaks faster than usual also dissent more than usual that year?
  by_session.png     per-session correlations for the main measures

Usage (needs the audio extra: uv sync --extra audio):
    uv run python audio_sessions.py --limit 3 --device mps          # quick check
    caffeinate -i uv run python audio_sessions.py --device mps      # everything, ~30 h
    # split across laptops: each runs one shard, then copy the speeches/ folders together
    caffeinate -i uv run python audio_sessions.py --device mps --shard 1/3
    uv run python audio_sessions.py --analyze-only
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import audio_tone as at
from un_data import DATA, load_processed

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "audio_sessions"
AUDIO_DIR = DATA / "audio_sessions"
BASE = "https://s3.amazonaws.com/downloads.unmultimedia.org/radio/library/ltd/mp3/ga"
SESSIONS = range(70, 81)
SR = at.SR

# Sessions 68-69 (2013-2014): no floor track, only one channel per UN language. The
# channel in the language the speaker used carries the speaker's own voice, so those
# speeches are usable when the language is known (see plan_old_sessions).
OLD_SESSIONS = (68, 69)
WSS = "https://s3.amazonaws.com/downloads.unmultimedia.org/wss"
STATEMENTS = "https://gadebate.un.org/sites/default/files/gastatements"
UN_LANGS_NOT_EN = ["fr", "es", "ar", "ru", "zh"]
OLD_PLAN = OUT / "old_sessions_plan.csv"

TRIM_START, TRIM_END = 30.0, 20.0     # seconds: President's introduction / thanks
VOICE_WINDOWS, VOICE_SECONDS = 24, 10.0
RATE_CLIPS, RATE_SECONDS = 6, 30.0
WHISPER_MODEL = "large-v3-turbo"

NO_SPACE_LANGS = {"th", "lo", "km", "my"}   # Thai, Lao, Khmer, Burmese
MIN_SPEECH_RATE = 0.5                         # words/s; below this the transcription failed
VOICE_COLS = ["arousal", "dominance", "valence"]
DELIVERY_COLS = ["pitch_mean", "pitch_sd", "loudness_db", "pause_share"]
MEASURES = ["text_valence"] + VOICE_COLS + DELIVERY_COLS + ["speech_rate", "speech_rate_in_lang"]

# Floor-audio file codes that differ from ISO 3166 alpha-2 (add any that show up as missing).
CODE_OVERRIDES = {}


# ---------------------------------------------------------------- corpus

def iso2(iso3):
    import pycountry
    if iso3 in CODE_OVERRIDES:
        return CODE_OVERRIDES[iso3]
    c = pycountry.countries.get(alpha_3=iso3)
    return c.alpha_2 if c else None


def audio_url(session, year, code):
    return f"{BASE}/{year}/{session}_{code}_FL.mp3"


def url_exists(url):
    """True if a GET for the first byte succeeds (the statements server refuses HEAD)."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.status in (200, 206)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError):
        return False


def statement_language(session, code):
    """Language of the official statement PDF: a statement in a UN language other than
    English means the speech was given in it; "en" if only English is found; else None."""
    c = code.lower()
    def found(lang):
        return any(url_exists(f"{STATEMENTS}/{session}/{name}") for name in
                   (f"{c}_{lang}_25.pdf", f"{session}_{c}_{lang}_25.pdf",
                    f"{c}_{lang}_24.pdf", f"{session}_{c}_{lang}_24.pdf"))
    for lang in UN_LANGS_NOT_EN:
        if found(lang):
            return lang
    return "en" if found("en") else None


def neighbour_languages():
    """Most common detected language per country in sessions 70-72 (from cached results)."""
    rows = [json.loads(p.read_text()) for p in (OUT / "speeches").glob("*.json")]
    t = pd.DataFrame([r for r in rows if r.get("session") in (70, 71, 72) and r.get("language")])
    if t.empty:
        return {}
    return t.groupby("iso3")["language"].agg(lambda x: x.mode().iat[0]).to_dict()


def plan_old_sessions(log=print):
    """Decide which channel carries the speaker's voice for each 2013-2014 speech.

    statement PDF in fr/es/ar/ru/zh -> that channel (source "statement")
    otherwise, the country spoke a UN language in 2015-2017 -> that channel
        (source "neighbour_years"; an assumption: the leader could have changed)
    otherwise (a language without a UN channel, such as Portuguese) -> skipped
    """
    speeches = load_processed()[0]
    s = speeches[speeches["session"].isin(OLD_SESSIONS) & (speeches["n_votes"] > 0)].copy()
    s["code"] = s["iso3"].map(iso2)
    s = s.dropna(subset=["code"])
    neighbours = neighbour_languages()
    rows = []
    for i, r in enumerate(s.itertuples(), 1):
        stmt = statement_language(r.session, r.code)
        if stmt in UN_LANGS_NOT_EN:
            lang, source = stmt, "statement"
        elif neighbours.get(r.iso3) in UN_LANGS_NOT_EN + ["en"]:
            lang, source = neighbours[r.iso3], "neighbour_years"
        else:
            lang, source = None, f"skipped: statement {stmt}, 2015-17 language {neighbours.get(r.iso3)}"
        rows.append({"speech_id": r.speech_id, "iso3": r.iso3, "session": int(r.session),
                     "statement_language": stmt, "neighbour_language": neighbours.get(r.iso3),
                     "channel_language": lang, "language_source": source,
                     "url": f"{WSS}/ga{r.session}/{lang}/{r.session}_{r.code}_{lang}.mp3" if lang else None})
        if i % 25 == 0:
            log(f"  planned {i}/{len(s)}")
    plan = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OLD_PLAN, index=False)
    use = plan.dropna(subset=["channel_language"])
    log(f"\n{len(use)} of {len(plan)} speeches from 2013-2014 usable")
    log("channel languages: " + ", ".join(f"{k} {v}" for k, v in use["channel_language"].value_counts().items()))
    log("decided by: " + ", ".join(f"{k} {v}" for k, v in use["language_source"].value_counts().items()))
    log(f"skipped: {len(plan) - len(use)} (no UN-language channel with the speaker's voice)")
    log(f"plan written to {OLD_PLAN}")
    return plan


def load_corpus(sessions=SESSIONS):
    """Speeches in `sessions` with their official text, floor-audio URL and votes."""
    speeches = load_processed()[0]
    s = speeches[speeches["session"].isin(list(sessions)) & (speeches["n_votes"] > 0)].copy()
    s["code"] = s["iso3"].map(iso2)
    s = s.dropna(subset=["code"])
    s["url"] = [audio_url(r.session, r.year, r.code) for r in s.itertuples()]
    old = s["session"].isin(OLD_SESSIONS)
    if old.any():
        if not OLD_PLAN.exists():
            raise SystemExit("run `audio_sessions.py --plan-old` first for sessions 68-69")
        plan = pd.read_csv(OLD_PLAN).dropna(subset=["channel_language"])
        plan = plan.set_index("speech_id")[["url", "channel_language", "language_source"]]
        keep = ~old | s["speech_id"].isin(plan.index)
        s = s[keep].copy()
        o = s["session"].isin(OLD_SESSIONS)
        for col in ["url", "channel_language", "language_source"]:
            if col not in s:
                s[col] = None
            s.loc[o, col] = s.loc[o, "speech_id"].map(plan[col])
    return s.sort_values(["session", "iso3"]).reset_index(drop=True)


def download(url, folder=AUDIO_DIR, tries=3):
    """Local path of the file at `url`, or None if the UN has no such file (403/404)."""
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / url.rsplit("/", 1)[1]
    if local.exists() and local.stat().st_size > 0:
        return local
    for attempt in range(tries):
        try:
            tmp = local.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(local)
            return local
        except urllib.error.HTTPError as err:
            if err.code in (403, 404):
                return None
            time.sleep(2 ** attempt)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------- measures

def trim(signal):
    """Drop the President's introduction and thanks when the speech is long enough."""
    start, end = int(TRIM_START * SR), len(signal) - int(TRIM_END * SR)
    return signal[start:end] if end - start > 60 * SR else signal


def spaced_clips(signal, n, seconds):
    """n clips of `seconds`, evenly spaced over the signal (fewer if it is short)."""
    size = int(seconds * SR)
    if len(signal) <= size:
        return [signal]
    n = max(1, min(n, len(signal) // size))
    starts = np.linspace(0, len(signal) - size, n).astype(int)
    return [signal[s:s + size] for s in starts]


def text_chunks(text, max_words=60):
    """The official text in chunks of whole sentences, up to max_words each."""
    sentences = re.split(r"(?<=[.!?])\s+", str(text).replace("\n", " "))
    chunks, current = [], []
    for sent in sentences:
        current.append(sent)
        if sum(len(x.split()) for x in current) >= max_words:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


def measure_speech(signal, text, voice, sentiment, whisper, lang=None):
    """All speech-level measures for one speech (signal already decoded at 16 kHz).
    `lang`, if given, is the known language of the recording (2013-2014 channels)."""
    body = trim(signal)
    windows = spaced_clips(body, VOICE_WINDOWS, VOICE_SECONDS)
    v = pd.DataFrame([voice(w) for w in windows]).mean().to_dict()
    d = at.delivery(body, n_words=0)
    d.pop("speech_rate", None)
    detected, share = whisper.language(body)
    lang = lang or detected
    words, seconds = 0, 0.0
    for clip in spaced_clips(body, RATE_CLIPS, RATE_SECONDS):
        segs, _ = whisper.model.transcribe(clip, language=lang, vad_filter=False)
        words += sum(at.count_units(s.text, lang) for s in segs)
        seconds += len(clip) / SR
    chunks = text_chunks(text)
    tv = float(np.mean(sentiment(chunks))) if chunks else np.nan
    return {**v, **d, "text_valence": tv, "language": lang, "language_share": share,
            "detected_language": detected,
            "speech_rate": words / seconds if seconds else np.nan,
            "minutes": len(signal) / SR / 60}


# ---------------------------------------------------------------- run

def process(corpus, voice, sentiment, whisper, redo=False, log=print, keep_audio=False):
    cache = OUT / "speeches"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [r for r in corpus.itertuples()
            if redo or not (cache / f"{r.speech_id}.json").exists()]
    log(f"{len(corpus)} speeches in this run, {len(corpus) - len(todo)} already done,"
        f" {len(todo)} to process")
    with ThreadPoolExecutor(max_workers=2) as pool:  # download the next file meanwhile
        futures = {i: pool.submit(download, r.url) for i, r in enumerate(todo[:2])}
        for i, r in enumerate(todo):
            t0 = time.time()
            path = futures.pop(i).result()
            if i + 2 < len(todo):
                futures[i + 2] = pool.submit(download, todo[i + 2].url)
            row = {"speech_id": r.speech_id, "iso3": r.iso3, "country": r.country,
                   "session": int(r.session), "year": int(r.year), "url": r.url}
            channel = getattr(r, "channel_language", None)
            if isinstance(channel, str):
                row["channel_language"] = channel
                row["language_source"] = r.language_source
            if path is None:
                row["missing"] = True
                log(f"  [{i + 1}/{len(todo)}] {r.speech_id} {r.country}: no floor audio")
            else:
                try:
                    row.update(measure_speech(at.load_audio(path), r.text, voice, sentiment, whisper,
                                              lang=row.get("channel_language")))
                except Exception as err:  # a broken file should not stop an overnight run
                    log(f"  [{i + 1}/{len(todo)}] {r.speech_id}: failed ({err}); will retry next run")
                    continue
                log(f"  [{i + 1}/{len(todo)}] {r.speech_id} {r.country} ({row['language']}):"
                    f" {row['minutes']:.0f} min, {time.time() - t0:.0f}s")
                if not keep_audio:  # ~9 MB per speech; the measures are saved below
                    path.unlink(missing_ok=True)
            (cache / f"{r.speech_id}.json").write_text(json.dumps(row))


# ---------------------------------------------------------------- language fix

# Languages a speech could plausibly be given in, per country: the six UN languages plus
# any language spoken by 25%+ of the country, or official and spoken by 15%+ (from the
# CLDR territory data via Babel; built once, see country_languages.json).
COUNTRY_LANGUAGES = ROOT / "country_languages.json"


def implausible(rows, allowed):
    """Speeches whose detected language is not plausible for the country."""
    return [r for r in rows if not r.get("missing") and "language" in r
            and r["language"] not in allowed.get(r["iso3"], [r["language"]])]


def redetect(whisper, body, allowed, n_clips=5):
    """Most likely allowed language, summing the full Whisper model's probabilities
    over evenly spaced clips (the main model, much better than the small detector)."""
    supported = set(getattr(whisper.model, "supported_languages", []) or allowed)
    probs = {}
    for clip in spaced_clips(body, n_clips, RATE_SECONDS):
        _, _, all_probs = whisper.model.detect_language(audio=clip)
        for lang, prob in all_probs:
            probs[lang] = probs.get(lang, 0.0) + prob
    candidates = [l for l in allowed if l in supported and l in probs]
    return max(candidates, key=probs.get) if candidates else "en"


def speech_rate_in(whisper, body, lang):
    words, seconds = 0, 0.0
    for clip in spaced_clips(body, RATE_CLIPS, RATE_SECONDS):
        segs, _ = whisper.model.transcribe(clip, language=lang, vad_filter=False)
        words += sum(at.count_units(x.text, lang) for x in segs)
        seconds += len(clip) / SR
    return words / seconds if seconds else float("nan")


def fix_languages(whisper, log=print, keep_audio=False):
    """Re-detect the language of implausible speeches and re-measure their speaking rate.
    Other measures do not depend on the language and are kept."""
    allowed = json.loads(COUNTRY_LANGUAGES.read_text())
    paths = sorted((OUT / "speeches").glob("*.json"))
    rows = {p: json.loads(p.read_text()) for p in paths}
    todo = [(p, r) for p, r in rows.items() if r in implausible([r], allowed)]
    log(f"{len(todo)} speeches with an implausible language to re-check")
    for i, (path_json, r) in enumerate(todo, 1):
        t0 = time.time()
        audio = download(r["url"])
        if audio is None:
            log(f"  [{i}/{len(todo)}] {r['speech_id']}: audio no longer available, skipped")
            continue
        body = trim(at.load_audio(audio))
        old = r["language"]
        new = redetect(whisper, body, allowed[r["iso3"]])
        r["language_first_detected"] = old
        r["language"] = new
        r["language_fixed"] = True
        r["speech_rate"] = speech_rate_in(whisper, body, new)
        path_json.write_text(json.dumps(r))
        if not keep_audio:
            audio.unlink(missing_ok=True)
        log(f"  [{i}/{len(todo)}] {r['speech_id']} {r['country']}: {old} -> {new},"
            f" {r['speech_rate']:.2f} words/s, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- analysis

def dissent_by_speech():
    """Share of No/Abstain among each country-session's votes on contested resolutions."""
    _, votes, resolutions = load_processed()
    v = votes.merge(resolutions[["undl_id", "contested"]], on="undl_id")
    v = v[v["contested"] & v["vote"].isin(["Y", "N", "A"])]
    return v["vote"].ne("Y").groupby(v["speech_id"]).mean().rename("dissent")


def load_tone():
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "speeches").glob("*.json"))]
    t = pd.DataFrame(rows)
    if "missing" in t:
        t = t[t["missing"].isna() | (t["missing"] == False)].drop(columns="missing")  # noqa: E712
    t = t.dropna(subset=["arousal"]).reset_index(drop=True)
    # Speaking rate is unusable for languages written without spaces between words (word
    # counts from splitting on spaces are meaningless; Chinese and Japanese are counted by
    # character instead) and below 0.5 words/s, which no real speech reaches: Whisper
    # transcribed the clips in the wrong language. Other measures of those speeches are kept.
    bad_rate = t["language"].isin(NO_SPACE_LANGS) | (t["speech_rate"] < MIN_SPEECH_RATE)
    t["speech_rate_excluded"] = bad_rate
    t.loc[bad_rate, "speech_rate"] = np.nan
    # speaking rate is only comparable within a language (languages with 5+ speeches)
    g = t.groupby("language")["speech_rate"]
    t["speech_rate_in_lang"] = ((t["speech_rate"] - g.transform("mean"))
                                / g.transform("std").replace(0, np.nan)).where(g.transform("size") >= 5)
    t["dissent"] = t["speech_id"].map(dissent_by_speech())
    return t.dropna(subset=["dissent"])


def _z(df, cols, by):
    g = df.groupby(by)[cols]
    return (df[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def spearman(a, b):
    ok = a.notna() & b.notna()
    return a[ok].corr(b[ok], method="spearman"), int(ok.sum())


def p_value(rho, n):
    from scipy import stats
    if n < 4 or not np.isfinite(rho):
        return np.nan
    t = rho * np.sqrt((n - 2) / max(1 - rho ** 2, 1e-12))
    return float(2 * stats.t.sf(abs(t), n - 2))


def pooled(t, n_boot=1000, seed=0):
    """Within-session standardised Spearman, with a country-resampling bootstrap CI."""
    z = _z(t, MEASURES + ["dissent"], "session").assign(iso3=t["iso3"].to_numpy())
    rng = np.random.default_rng(seed)
    countries = z["iso3"].unique()
    groups = {c: idx for c, idx in z.groupby("iso3").indices.items()}
    rows = []
    for col in MEASURES:
        rho, n = spearman(z[col], z["dissent"])
        boots = []
        for _ in range(n_boot):
            idx = np.concatenate([groups[c] for c in rng.choice(countries, len(countries))])
            b = z.iloc[idx]
            boots.append(spearman(b[col], b["dissent"])[0])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append({"measure": col, "spearman": rho, "ci95_low": lo, "ci95_high": hi,
                     "excludes_zero": bool(lo > 0 or hi < 0), "n_speeches": n,
                     "n_countries": int(z.loc[z[col].notna(), "iso3"].nunique())})
    return pd.DataFrame(rows)


def by_session(t):
    rows = []
    for s, d in t.groupby("session"):
        for col in MEASURES:
            rho, n = spearman(d[col], d["dissent"])
            rows.append({"session": s, "year": int(d["year"].iloc[0]), "measure": col,
                         "spearman": rho, "p": p_value(rho, n), "n": n})
    return pd.DataFrame(rows)


def country_level(t):
    z = _z(t, MEASURES + ["dissent"], "session").assign(iso3=t["iso3"].to_numpy())
    c = z.groupby("iso3").mean()
    rows = []
    for col in MEASURES:
        rho, n = spearman(c[col], c["dissent"])
        rows.append({"measure": col, "spearman": rho, "p": p_value(rho, n), "n_countries": n})
    return pd.DataFrame(rows)


def within_country(t):
    """Deviations from each country's own average (countries with 3+ sessions)."""
    z = _z(t, MEASURES + ["dissent"], "session").assign(iso3=t["iso3"].to_numpy())
    z = z[z.groupby("iso3")["iso3"].transform("size") >= 3]
    dev = z[MEASURES + ["dissent"]] - z.groupby("iso3")[MEASURES + ["dissent"]].transform("mean")
    rows = []
    for col in MEASURES:
        rho, n = spearman(dev[col], dev["dissent"])
        rows.append({"measure": col, "spearman": rho, "p": p_value(rho, n), "n_speeches": n,
                     "n_countries": int(z["iso3"].nunique())})
    return pd.DataFrame(rows)


def plot_by_session(bs, path, measures=("speech_rate_in_lang", "text_valence", "arousal")):
    fig, ax = at._axes("Correlation with dissent on contested votes, by session",
                       "year", "Spearman ρ (within session)")
    colors = [at.BLUE, at.ORANGE, at.INK2]
    for m, color in zip(measures, colors):
        d = bs[bs["measure"] == m].sort_values("year")
        ax.plot(d["year"], d["spearman"], marker="o", color=color, label=m.replace("_", " "))
    ax.axhline(0, color=at.INK2, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=at.INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=at.SURFACE)


def analyze(log=print):
    import warnings
    with warnings.catch_warnings():
        # resamples or sessions where a measure is constant give undefined correlations
        # (NaN); scipy and numpy warn about each one, which buries the results
        warnings.simplefilter("ignore", RuntimeWarning)
        return _analyze(log)


def _analyze(log=print):
    t = load_tone()
    outputs = {"tone": t, "pooled": pooled(t), "by_session": by_session(t),
               "country_level": country_level(t), "within_country": within_country(t)}
    for name, df in outputs.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    plot_by_session(outputs["by_session"], OUT / "by_session.png")
    log(f"\nspeaking rate excluded for {int(t['speech_rate_excluded'].sum())} speeches"
        f" (languages without word spaces, or under {MIN_SPEECH_RATE} words/s)")
    if "language_fixed" in t:
        log(f"\nlanguage re-detected for {int(t['language_fixed'].fillna(False).sum())} speeches")
    log(f"\n{len(t)} speeches from {t['iso3'].nunique()} countries, sessions"
        f" {t['session'].min()}-{t['session'].max()}"
        f" ({', '.join(f'{s}: {n}' for s, n in t.groupby('session').size().items())})")
    log("\nPooled over sessions (within-session standardised; CI resamples countries):")
    log(outputs["pooled"].round(3).to_string(index=False))
    log("\nCountry level (each country's average over the years):")
    log(outputs["country_level"].round(3).to_string(index=False))
    log("\nWithin country (deviations from the country's own average):")
    log(outputs["within_country"].round(3).to_string(index=False))
    log(f"\nwritten to {OUT}")
    return outputs


def parse_sessions(text):
    if "-" in text:
        a, b = text.split("-")
        return range(int(a), int(b) + 1)
    return [int(x) for x in text.split(",")]


def main(sessions=SESSIONS, limit=None, device="cpu", redo=False, shard=None,
         whisper_model=WHISPER_MODEL, keep_audio=False):
    OUT.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus(sessions)
    if shard:
        k, n = (int(x) for x in shard.split("/"))
        corpus = corpus.iloc[k - 1::n].reset_index(drop=True)
        print(f"shard {k}/{n}")
    if limit:
        corpus = corpus.head(limit)
    print("loading models (first run downloads them)")
    whisper = at.Whisper(whisper_model, device)
    process(corpus, at.VoiceEmotion(device=device), at.TextSentiment(at.TEXT_MODEL_EN, device=device),
            whisper, redo=redo, keep_audio=keep_audio)
    return analyze()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sessions", default="70-80", help="e.g. 70-80 (default) or 75,80")
    p.add_argument("--limit", type=int, help="only the first N speeches")
    p.add_argument("--device", default="cpu", help="cpu or mps (Apple silicon)")
    p.add_argument("--shard", help="K/N: process every Nth speech starting at K (split across machines)")
    p.add_argument("--whisper-model", default=WHISPER_MODEL)
    p.add_argument("--redo", action="store_true", help="recompute cached speeches")
    p.add_argument("--keep-audio", action="store_true",
                   help="keep the downloaded mp3s (by default each is deleted once measured)")
    p.add_argument("--analyze-only", action="store_true", help="only rerun the analysis")
    p.add_argument("--plan-old", action="store_true",
                   help="work out which channel carries each 2013-2014 speaker's voice"
                        " (writes results/audio_sessions/old_sessions_plan.csv)")
    p.add_argument("--fix-languages", action="store_true",
                   help="re-detect implausible languages (see country_languages.json), re-measure"
                        " their speaking rate, then rerun the analysis")
    a = p.parse_args()
    if a.analyze_only:
        OUT.mkdir(parents=True, exist_ok=True)
        analyze()
    elif a.plan_old:
        plan_old_sessions()
    elif a.fix_languages:
        print("loading Whisper")
        fix_languages(at.Whisper(a.whisper_model, a.device), keep_audio=a.keep_audio)
        analyze()
    else:
        main(parse_sessions(a.sessions), a.limit, a.device, a.redo, a.shard, a.whisper_model,
             a.keep_audio)
