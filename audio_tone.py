"""Audio tone case study: does how a leader *sounds* differ from what the words *say*?

Data: the UN transcription corpus on Hugging Face (united-nations/transcription-corpus,
"gadebate" subset, CC BY 4.0): the General Debate of the 80th session (2025), one row
per country, with the floor audio (the speaker's own voice) and each interpretation.
The floor audio is always used, so the voice is the leader's own, never an interpreter's.

Two modes (--languages):
  all  (default) every speech, in whatever language it was given (192 rows). Whisper
       transcribes in the speaker's language and text sentiment uses a multilingual model.
       Results go to results/audio_all/.
  en   only speeches delivered in English, with English-only models: the original
       74-speech study. Results go to results/audio/.

For each speech:
  1. download the floor audio (cached in data/audio/)
  2. detect the speech's language with Whisper (the corpus language field is not always
     right), transcribe it in that language (in "en" mode, speeches that are not English
     are skipped), and merge its segments into passages of 5-20 seconds that end on a
     sentence boundary where possible. Passages not in English are machine-translated
     (opus-mt-mul-en) into text_en, used only to tag issues with the English keywords.
     In a speech that is not in English, passages
     Whisper hears in another language are dropped: they are mostly the Assembly President
     introducing or thanking the speaker in English
  3. score each passage twice:
       text   sentiment of the words: P(positive) - P(negative), from -1 to 1
       voice  arousal, dominance and valence of the voice (audeering wav2vec2 model
              fine-tuned on MSP-Podcast; roughly 0 to 1)
     plus delivery features: pitch level and variation (semitones), loudness (dB),
     share of pauses, speaking rate (words per second; characters per second for
     Chinese and Japanese, which are written without spaces)
     Caveat: the voice model was trained on English speech (MSP-Podcast), so its scores
     are less reliable for speeches in other languages. Speaking rate is not comparable
     across languages, so the comparison with votes also uses speech_rate_in_lang, the
     rate standardised among speeches in the same language.
     Caveat: the voice model's *valence* is known to draw partly on what is being said
     (its authors report it is sensitive to linguistic content), so a correlation
     between text sentiment and vocal valence overstates how much the voice alone
     agrees with the words. Arousal and dominance are mostly acoustic.
  4. z-score every measure within the speech, so passages are compared with the same
     speaker's average (this removes differences in voice, microphone and style)

Then, across speeches:
  - text vs voice: how closely do text sentiment and vocal valence / arousal track each
    other, passage by passage?
  - mismatches: the passages where the words and the voice disagree most
  - tone by issue: passages tagged with the issue keywords from un_data.py
  - tone and votes: each country's average tone vs its share of No / Abstain votes on
    contested resolutions that session (descriptive only), for all speeches and for the
    English-delivered ones alone

Outputs go to results/audio_all/ (or results/audio/ with --languages en): passages.csv, speeches.csv, mismatches.csv,
issues.csv, correlations.csv and three PNG figures.

Usage (needs the audio extra: uv sync --extra audio):
    uv run python audio_tone.py --limit 3             # quick check on 3 speeches
    uv run python audio_tone.py --device mps           # all 192 speeches, every language
    uv run python audio_tone.py --languages en         # the original English-only study
    uv run python audio_tone.py --whisper-model medium --device mps
Per-speech results are cached in <output folder>/passages/, so a stopped run resumes
where it left off (--redo recomputes them; --analyze-only reruns just the analysis).
In "all" mode, English speeches already processed by the English-only study are reused
from results/audio/passages/ (their transcripts and voice scores are kept; only the text
sentiment is rescored with the multilingual model), so they are not transcribed again.
"""

import argparse
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from un_data import DATA, ISSUE_PATTERNS, ISSUES, load_split

ROOT = Path(__file__).parent
OUT_EN = ROOT / "results" / "audio"
OUT_ALL = ROOT / "results" / "audio_all"
OUT = OUT_ALL  # set by main() / --languages
AUDIO_DIR = DATA / "audio"

REPO = "united-nations/transcription-corpus"
SUBSET = "gadebate"
SESSION = 80
SR = 16_000

WHISPER_MODEL = "small"            # multilingual
WHISPER_MODEL_EN = "small.en"      # English-only mode
EMOTION_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"         # multilingual
TEXT_MODEL_EN = "cardiffnlp/twitter-roberta-base-sentiment-latest"   # English-only mode
# Issue tags use the English keyword patterns from un_data, so passages in other languages
# are machine-translated to English (text_en) for tagging only; sentiment uses the original.
TRANSLATE_MODEL = "Helsinki-NLP/opus-mt-mul-en"

# Sentence-final punctuation, including Chinese/Japanese full-width marks and the Arabic
# question mark, so passages can end on a sentence boundary in any language.
SENTENCE_END = (".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u061f")
CHARACTER_LANGS = {"zh", "ja"}  # written without spaces: count characters, not words

MIN_SECONDS, MAX_SECONDS = 5.0, 20.0
TEXT_COLS = ["text_valence"]
VOICE_COLS = ["arousal", "dominance", "valence"]
DELIVERY_COLS = ["pitch_mean", "pitch_sd", "loudness_db", "pause_share", "speech_rate"]
MEASURES = TEXT_COLS + VOICE_COLS + DELIVERY_COLS


# ---------------------------------------------------------------- corpus

def load_corpus():
    """The 2025 General Debate rows, with a speech_id that joins to un_data."""
    from datasets import load_dataset
    df = load_dataset(REPO, SUBSET, split="train").to_pandas()
    df["speech_id"] = [to_speech_id(iso, name) for iso, name
                       in zip(df["country_iso"], df["country_name"])]
    df["english"] = df["original_lang"].map(is_english)
    return df


def to_speech_id(iso2, name=None, session=SESSION):
    """'AL' -> 'ALB_80'. Falls back to the country name; None when neither matches."""
    import pycountry
    country = None
    if isinstance(iso2, str) and iso2.strip():
        country = pycountry.countries.get(alpha_2=iso2.strip().upper())
    if country is None and isinstance(name, str):
        try:
            country = pycountry.countries.lookup(name)
        except LookupError:
            country = None
    return f"{country.alpha_3}_{session}" if country else None


def is_english(lang):
    return isinstance(lang, str) and lang.strip().lower() in {"en", "eng", "english"}


def count_units(text, lang="en"):
    """Words in `text`; for Chinese and Japanese, non-space characters (no spaces between words)."""
    if lang in CHARACTER_LANGS:
        return sum(1 for ch in text if not ch.isspace() and ch.isalnum())
    return len(text.split())


def fetch_audio(ref, folder=AUDIO_DIR):
    """Local path of an audio file given its URL or its path inside the dataset repo."""
    folder.mkdir(parents=True, exist_ok=True)
    local = folder / Path(ref).name
    if local.exists():
        return local
    match = re.search(r"/resolve/[^/]+/(.+)$", ref)
    if match or not ref.startswith("http"):
        from huggingface_hub import hf_hub_download
        path = match.group(1) if match else ref.lstrip("/")
        return Path(hf_hub_download(REPO, path, repo_type="dataset", local_dir=folder.parent / "hf"))
    urllib.request.urlretrieve(ref, local)
    return local


# ---------------------------------------------------------------- passages

def merge_segments(segments, min_s=MIN_SECONDS, max_s=MAX_SECONDS):
    """Merge Whisper segments ({start, end, text}) into passages of min_s-max_s seconds.

    A passage closes once it is at least min_s long and ends a sentence, or once it
    reaches max_s. A short leftover at the end joins the previous passage.
    """
    passages, current = [], None
    for seg in segments:
        if current is None:
            current = dict(seg)
        else:
            current["end"] = seg["end"]
            current["text"] = f"{current['text'].strip()} {seg['text'].strip()}"
        length = current["end"] - current["start"]
        if (length >= min_s and current["text"].rstrip().endswith(SENTENCE_END)) or length >= max_s:
            passages.append(current)
            current = None
    if current is not None:
        if passages and current["end"] - current["start"] < min_s:
            passages[-1]["end"] = current["end"]
            passages[-1]["text"] = f"{passages[-1]['text'].strip()} {current['text'].strip()}"
        else:
            passages.append(current)
    for p in passages:
        p["text"] = p["text"].strip()
    return passages


# The floor audio also carries the Assembly President introducing and thanking each
# speaker, in a different voice (often over applause). Those passages are dropped
# before scoring: compared with the leader's own average, they look extreme.
CHAIR_PATTERN = (r"give the floor|request protocol|escort (?:his|her) excellency"
                 r"|will hear an? (?:address|statement)|i would like to introduce"
                 r"|on behalf of the general assembly|from the rostrum"
                 r"|for the statement (?:he|she) has just made|^\W*the floor\W*$")


def is_chair(texts):
    """True for passages spoken by the Assembly President (introductions, thanks)."""
    return pd.Series(list(texts)).str.lower().str.contains(CHAIR_PATTERN, regex=True).to_numpy()


def issue_tags(texts):
    """Boolean issue_<area> columns for each passage (same keywords as un_data).

    Pass English text: for passages in other languages, use the text_en column."""
    lower = pd.Series(list(texts)).str.lower()
    return pd.DataFrame({f"issue_{k}": lower.str.contains(p, regex=True)
                         for k, p in ISSUE_PATTERNS.items()})


# ---------------------------------------------------------------- models
# Each scorer is a plain callable, so tests can swap in stand-ins.

class Whisper:
    """faster-whisper transcription, after a language check.

    Returns (segments, detected language, share of checked clips in that language).
    The corpus language field is not always right, and forcing English on a speech given
    in another language yields a garbled transcript. The check uses a small multilingual
    model (English-only models such as small.en always report English) on three 30-second
    clips from inside the speech: the first seconds are usually the Assembly President
    introducing the speaker, in English, whatever language the speaker then uses.
    """

    DETECT_MODEL = "base"
    CLIPS_AT = (0.25, 0.5, 0.75)

    def __init__(self, name=WHISPER_MODEL, device="cpu", english_only=False):
        from faster_whisper import WhisperModel
        if name.endswith(".en") and not english_only:
            raise ValueError(f"{name} only transcribes English; use a multilingual model"
                             " (e.g. small) or --languages en")
        self.english_only = english_only
        try:  # Apple-silicon (MPS) backend is experimental in CTranslate2; fall back to CPU
            self.model = WhisperModel(name, device=device,
                                      compute_type="float16" if device == "mps" else "int8")
        except (ValueError, RuntimeError) as err:
            print(f"  Whisper could not use {device} ({err}); using cpu")
            self.model = WhisperModel(name, device="cpu", compute_type="int8")
        self.detector = WhisperModel(self.DETECT_MODEL, device="cpu", compute_type="int8")

    def language(self, signal):
        """Majority language over clips at CLIPS_AT, and the share of clips agreeing."""
        clip = 30 * SR
        votes = []
        for at in self.CLIPS_AT:
            start = max(0, min(int(at * len(signal)) - clip // 2, len(signal) - clip))
            lang, _, _ = self.detector.detect_language(audio=signal[start:start + clip])
            votes.append(lang)
        top = max(set(votes), key=votes.count)
        return top, votes.count(top) / len(votes)

    def language_of(self, clip):
        """Language Whisper hears in one short clip (used to drop off-language passages)."""
        lang, _, _ = self.detector.detect_language(audio=clip)
        return lang

    def __call__(self, path, signal=None):
        signal = load_audio(path) if signal is None else signal
        lang, share = self.language(signal)
        if lang != "en" and self.english_only:
            return [], lang, share
        segments, _ = self.model.transcribe(signal, language=lang, vad_filter=True)
        return [{"start": s.start, "end": s.end, "text": s.text} for s in segments], lang, share


def emotion_model_class():
    """The model class from audeering's model card: wav2vec2, mean-pooled, regression head."""
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.classifier(hidden)

    return EmotionModel


class VoiceEmotion:
    """Arousal, dominance, valence from audeering's wav2vec2 model (outputs in that order).

    A clip is scored in windows of at most WINDOW_SECONDS and the windows are averaged
    (weighted by length): the model's memory grows quickly with clip length, and a long
    passage can exhaust GPU memory. If the GPU still runs out, the model moves to the
    CPU for the rest of the run.
    """

    WINDOW_SECONDS = 10.0

    def __init__(self, name=EMOTION_MODEL, device="cpu"):
        import torch
        from transformers import Wav2Vec2FeatureExtractor
        self.torch, self.device = torch, device
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
        self.model = emotion_model_class().from_pretrained(name).to(device).eval()

    def windows(self, signal):
        size = int(self.WINDOW_SECONDS * SR)
        n = max(1, int(np.ceil(len(signal) / size)))
        return [w for w in np.array_split(signal, n) if len(w) >= SR // 2]

    def _score(self, window):
        x = self.extractor(window, sampling_rate=SR, return_tensors="pt").input_values
        with self.torch.no_grad():
            out = self.model(x.to(self.device))[0].cpu().numpy()
        if self.device == "mps":
            self.torch.mps.empty_cache()
        return out

    def __call__(self, signal):
        scores, weights = [], []
        for w in self.windows(signal):
            try:
                scores.append(self._score(w))
            except (RuntimeError, MemoryError) as err:  # torch.OutOfMemoryError is a RuntimeError
                if self.device == "cpu" or "memory" not in str(err).lower():
                    raise
                print(f"  voice model ran out of {self.device} memory; moving it to cpu")
                self.device = "cpu"
                self.model.to("cpu")
                if self.torch.backends.mps.is_available():
                    self.torch.mps.empty_cache()
                scores.append(self._score(w))
            weights.append(len(w))
        arousal, dominance, valence = np.average(np.array(scores), axis=0, weights=weights)
        return {"arousal": float(arousal), "dominance": float(dominance), "valence": float(valence)}


class TextSentiment:
    """P(positive) - P(negative) for each text."""

    def __init__(self, name=TEXT_MODEL, device="cpu"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = name
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        if "positive" not in labels:  # generic LABEL_0/1/2 names: cardiffnlp order is neg, neu, pos
            labels = {"negative": 0, "positive": self.model.config.num_labels - 1}
        self.pos, self.neg = labels["positive"], labels["negative"]

    def __call__(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.model(**batch).logits.softmax(-1).cpu().numpy()
            out += (p[:, self.pos] - p[:, self.neg]).tolist()
        return out


class Translator:
    """Machine translation to English (Marian opus-mt-mul-en), used only for issue tags."""

    def __init__(self, name=TRANSLATE_MODEL, device="cpu"):
        import torch
        from transformers import MarianMTModel, MarianTokenizer
        self.torch, self.device = torch, device
        self.tokenizer = MarianTokenizer.from_pretrained(name)
        self.model = MarianMTModel.from_pretrained(name).to(device).eval()

    def __call__(self, texts, batch_size=16):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = self.tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                   max_length=256, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                ids = self.model.generate(**batch, max_new_tokens=256)
            out += self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        return out


def load_audio(path):
    from faster_whisper.audio import decode_audio
    return decode_audio(str(path), sampling_rate=SR)


def delivery(signal, n_words):
    """Pitch level and variation (semitones re 27.5 Hz), loudness, pauses, speaking rate."""
    import librosa
    frame, hop = 1024, 256
    rms = librosa.feature.rms(y=signal, frame_length=frame, hop_length=hop)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-6))
    voiced = db > np.median(db) - 20  # quieter than 20 dB below typical = pause
    f0 = librosa.yin(signal, fmin=65, fmax=400, sr=SR, frame_length=frame, hop_length=hop)
    n = min(len(f0), len(voiced))
    semitones = 12 * np.log2(f0[:n][voiced[:n]] / 27.5)
    seconds = len(signal) / SR
    return {"pitch_mean": float(np.mean(semitones)) if len(semitones) else np.nan,
            "pitch_sd": float(np.std(semitones)) if len(semitones) else np.nan,
            "loudness_db": float(np.mean(db[voiced])) if voiced.any() else np.nan,
            "pause_share": float(1 - voiced.mean()),
            "speech_rate": n_words / seconds if seconds else np.nan}


# ---------------------------------------------------------------- per speech

def score_speech(signal, segments, voice, text, lang="en", keep=None):
    """One row per passage: timing, text, text sentiment, voice and delivery measures.

    `keep(clip)`, if given, decides whether a passage is scored (used to drop passages
    in a different language from the speech, such as the President's introduction).
    """
    passages = merge_segments(segments)
    rows = []
    for p in passages:
        clip = signal[int(p["start"] * SR):int(p["end"] * SR)]
        if len(clip) < SR:  # under a second of audio: nothing to measure
            continue
        if keep is not None and not keep(clip):
            continue
        n_words = count_units(p["text"], lang)
        rows.append({"start": round(p["start"], 2), "end": round(p["end"], 2),
                     "text": p["text"], **voice(clip), **delivery(clip, n_words)})
    df = pd.DataFrame(rows)
    if len(df):
        df.insert(3, "text_valence", text(df["text"].tolist()))
    return df


def zscore_within(df, cols, by="speech_id"):
    g = df.groupby(by)[cols]
    return (df[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def _rescore_text(cached, text, text_name):
    """Cached passages scored with another text model: rescore only the text sentiment."""
    have = cached["text_model"].iloc[0] if "text_model" in cached else TEXT_MODEL_EN
    if have != text_name:
        cached["text_valence"] = text(cached["text"].fillna("").tolist())
        cached["text_model"] = text_name
    return cached


def _add_text_en(df, translate, lang):
    """text_en: the passage in English, for issue tagging (a copy of text if already English)."""
    if "text_en" in df and df["text_en"].notna().all():
        return df
    texts = df["text"].fillna("").tolist()
    df["text_en"] = texts if lang == "en" or translate is None else translate(texts)
    return df


def process(corpus, transcribe, voice, text, redo=False, log=print, english_only=False,
            reuse=None, translate=None):
    """Score every speech in `corpus` (cached per speech). Returns all passages.

    english_only: speeches Whisper detects as not English are skipped and listed in
    skipped.csv. Otherwise every speech is transcribed in its own language.
    reuse: another passages folder to take already-processed speeches from (their text
    sentiment is rescored if it came from a different text model).
    translate: callable list[str] -> list[str] into English, for issue tags of passages in
    other languages (stored as text_en). None keeps text_en = text.
    A cached file from before the language check was fixed is recomputed.
    """
    cache = OUT / "passages"
    cache.mkdir(parents=True, exist_ok=True)
    text_name = getattr(text, "name", TEXT_MODEL_EN if english_only else TEXT_MODEL)
    frames, skipped = [], []
    for i, row in enumerate(corpus.itertuples(), 1):
        path = cache / f"{row.speech_id}.csv"
        if not redo:
            source = path if path.exists() else (
                reuse / path.name if reuse is not None and (reuse / path.name).exists() else None)
            cached = pd.read_csv(source) if source is not None else None
            if cached is not None and "detected_language" in cached:
                if len(cached):
                    before = cached.get("text_model")
                    had_en = "text_en" in cached
                    cached = _rescore_text(cached, text, text_name)
                    lang = str(cached["detected_language"].iloc[0])
                    cached = _add_text_en(cached, translate, lang)
                    if (source != path or before is None or not had_en
                            or not cached["text_model"].equals(before)):
                        cached.to_csv(path, index=False)
                    frames.append(cached)
                    continue
                if english_only:
                    skipped.append({"speech_id": row.speech_id, "country": row.country_name,
                                    "detected_language": "not en (cached)"})
                    continue
                # skipped by an English-only run: process it now, in its own language
        t0 = time.time()
        audio = fetch_audio(row.audio_floor)
        signal = load_audio(audio)
        t1 = time.time()
        segments, lang, share = transcribe(audio, signal)
        t2 = time.time()
        if lang != "en" and english_only:
            log(f"  [{i}/{len(corpus)}] {row.speech_id} {row.country_name}: skipped, Whisper"
                f" hears {lang!r} in {share:.0%} of checked clips, not English")
            pd.DataFrame(columns=["speech_id", "country", "detected_language"]).to_csv(path, index=False)
            skipped.append({"speech_id": row.speech_id, "country": row.country_name,
                            "detected_language": lang, "share_of_clips": share})
            continue
        dropped = [0]
        keep = None
        if lang != "en" and hasattr(transcribe, "language_of"):
            def keep(clip, lang=lang):
                ok = transcribe.language_of(clip) == lang
                dropped[0] += not ok
                return ok
        df = score_speech(signal, segments, voice, text, lang=lang, keep=keep)
        if not len(df):
            log(f"  [{i}/{len(corpus)}] {row.speech_id} {row.country_name}: no passages"
                f" in {lang!r}; skipped")
            skipped.append({"speech_id": row.speech_id, "country": row.country_name,
                            "detected_language": lang, "share_of_clips": share})
            continue
        df.insert(0, "speech_id", row.speech_id)
        df.insert(1, "country", row.country_name)
        df["detected_language"] = lang
        df["text_model"] = text_name
        df = _add_text_en(df, translate, lang)
        df.to_csv(path, index=False)
        frames.append(df)
        minutes = len(signal) / SR / 60
        extra = f", {dropped[0]} off-language passages dropped" if keep is not None else ""
        log(f"  [{i}/{len(corpus)}] {row.speech_id} {row.country_name} ({lang}): {len(df)} passages"
            f"{extra}, {minutes:.0f} min of audio | download {t1 - t0:.0f}s, transcribe"
            f" {t2 - t1:.0f}s, score {time.time() - t2:.0f}s")
    if skipped:
        pd.DataFrame(skipped).to_csv(OUT / "skipped.csv", index=False)
        log(f"  skipped {len(skipped)} speeches ({OUT.relative_to(ROOT)}/skipped.csv)")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------- analysis

def add_z_and_issues(passages):
    z = zscore_within(passages, MEASURES).add_prefix("z_")
    english = (passages["text_en"].fillna(passages["text"]) if "text_en" in passages
               else passages["text"])
    return pd.concat([passages.reset_index(drop=True), z.reset_index(drop=True),
                      issue_tags(english)], axis=1)


def correlations(passages):
    """Within-speech correlation of text sentiment with each voice measure (pooled z)."""
    rows = []
    for col in VOICE_COLS + DELIVERY_COLS:
        a, b = passages["z_text_valence"], passages[f"z_{col}"]
        ok = a.notna() & b.notna()
        per_speech = passages[ok].groupby("speech_id").apply(
            lambda d: d["text_valence"].corr(d[col]), include_groups=False)
        rows.append({"voice_measure": col, "pooled_r": a[ok].corr(b[ok]),
                     "median_speech_r": per_speech.median(),
                     "share_speeches_r_positive": (per_speech > 0).mean(),
                     "n_passages": int(ok.sum()), "n_speeches": int(per_speech.notna().sum())})
    return pd.DataFrame(rows)


def mismatches(passages, n=15):
    """Passages where words and voice disagree most (in within-speech SDs)."""
    d = passages.assign(gap=passages["z_text_valence"] - passages["z_valence"]).dropna(subset=["gap"])
    cols = ["speech_id", "country", "start", "end", "gap", "z_text_valence", "z_valence",
            "z_arousal", "text"]
    words_warmer = d.nlargest(n, "gap")[cols].assign(kind="words more positive than voice")
    voice_warmer = d.nsmallest(n, "gap")[cols].assign(kind="voice more positive than words")
    intense = (d[d["z_text_valence"].abs() < 0.5].nlargest(n, "z_arousal")[cols]
               .assign(kind="neutral words, most intense delivery"))
    return pd.concat([words_warmer, voice_warmer, intense], ignore_index=True).round(3)


def by_issue(passages):
    """Mean within-speech z of each measure for passages tagged with each issue."""
    rows = []
    for issue in ISSUES:
        d = passages[passages[f"issue_{issue}"]]
        for col in ["text_valence", "valence", "arousal", "dominance"]:
            v = d[f"z_{col}"].dropna()
            se = v.std() / np.sqrt(len(v)) if len(v) > 1 else np.nan
            rows.append({"issue": issue, "measure": col, "mean_z": v.mean(),
                         "ci95": 1.96 * se, "n_passages": len(v)})
    return pd.DataFrame(rows)


def speech_summary(passages):
    """Average raw tone per speech, joined to the country's 2025 votes."""
    s = passages.groupby(["speech_id", "country"])[MEASURES].mean().reset_index()
    s["n_passages"] = passages.groupby("speech_id").size().reindex(s["speech_id"]).to_numpy()
    lang = (passages.groupby("speech_id")["detected_language"].agg(lambda x: x.mode().iat[0])
            if "detected_language" in passages else pd.Series(dtype=object))
    s["language"] = s["speech_id"].map(lang).fillna("en")
    # Words per second differ by language, so also standardise speaking rate among speeches
    # in the same language (languages with fewer than 3 speeches get NaN).
    g = s.groupby("language")["speech_rate"]
    big = g.transform("size") >= 3
    s["speech_rate_in_lang"] = ((s["speech_rate"] - g.transform("mean"))
                                / g.transform("std").replace(0, np.nan)).where(big)
    votes = pd.concat([load_split("test")[1], load_split("test_heldout")[1]])
    votes = votes[votes["session"] == SESSION]
    contested = votes[votes["contested"]]
    dissent = contested["vote"].ne("Y").groupby(contested["speech_id"]).mean()
    s["dissent_contested"] = s["speech_id"].map(dissent)
    s["n_votes"] = s["speech_id"].map(votes.groupby("speech_id").size())
    return s


def tone_vs_votes(speeches):
    """Spearman correlation of each speech-level measure with dissent, for all speeches
    and for English-delivered speeches alone."""
    groups = {"all": speeches}
    if "language" in speeches and (speeches["language"] != "en").any():
        groups["english_only"] = speeches[speeches["language"] == "en"]
    measures = MEASURES + (["speech_rate_in_lang"] if "speech_rate_in_lang" in speeches else [])
    rows = []
    for group, d in groups.items():
        for col in measures:
            ok = d[[col, "dissent_contested"]].dropna()
            rows.append({"speeches": group, "measure": col, "spearman_with_dissent":
                         ok[col].corr(ok["dissent_contested"], method="spearman"),
                         "n_countries": len(ok)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"


def _axes(title, xlabel, ylabel, size=(6.4, 4.4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    return fig, ax


def plot_text_vs_voice(passages, path):
    d = passages[["z_text_valence", "z_valence"]].dropna()
    r = d.corr().iloc[0, 1]
    fig, ax = _axes(f"Words vs voice, passage by passage (r = {r:.2f})",
                    "text sentiment (SDs from the speaker's average)",
                    "vocal valence (SDs from the speaker's average)")
    ax.scatter(d["z_text_valence"], d["z_valence"], s=10, color=BLUE, alpha=0.25,
               linewidths=0)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.axvline(0, color=INK2, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def plot_issues(issues, path):
    d = issues[issues["measure"].isin(["text_valence", "arousal"]) & (issues["n_passages"] > 1)]
    order = (d[d["measure"] == "arousal"].sort_values("mean_z")["issue"].tolist())
    fig, ax = _axes("Tone by issue (passages tagged by keyword)",
                    "mean, in SDs from the speaker's average (95% CI)", "")
    for offset, (measure, color, label) in zip(
            [-0.15, 0.15], [("text_valence", ORANGE, "text sentiment"),
                            ("arousal", BLUE, "vocal arousal")]):
        m = d[d["measure"] == measure].set_index("issue").loc[order]
        y = np.arange(len(order)) + offset
        ax.errorbar(m["mean_z"], y, xerr=m["ci95"], fmt="o", color=color, ms=6,
                    elinewidth=1.5, capsize=0, label=label)
    ax.set_yticks(np.arange(len(order)), [o.replace("_", " ") for o in order])
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


def plot_tone_vs_votes(speeches, path, measure="arousal", n_labels=6):
    d = speeches[["speech_id", measure, "dissent_contested"]].dropna()
    rho = d[measure].corr(d["dissent_contested"], method="spearman")
    fig, ax = _axes(f"Vocal {measure} vs voting against the majority "
                    f"(Spearman ρ = {rho:.2f}, n = {len(d)})",
                    f"mean vocal {measure} over the speech",
                    "share of No / Abstain on contested resolutions, 2025")
    ax.scatter(d[measure], d["dissent_contested"], s=40, color=BLUE, alpha=0.8,
               edgecolors=SURFACE, linewidths=1.5)
    extremes = pd.concat([d.nlargest(n_labels // 2, "dissent_contested"),
                          d.nlargest(n_labels // 2, measure)]).drop_duplicates("speech_id")
    for r in extremes.itertuples():
        ax.annotate(r.speech_id.split("_")[0], (getattr(r, measure), r.dissent_contested),
                    xytext=(5, 3), textcoords="offset points", fontsize=7, color=INK2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)


# ---------------------------------------------------------------- run

def analyze(passages, log=print):
    chair = is_chair(passages["text"])
    log(f"dropped {chair.sum()} passages spoken by the Assembly President")
    passages = add_z_and_issues(passages[~chair].reset_index(drop=True))
    speeches = speech_summary(passages)
    outputs = {"passages": passages, "speeches": speeches,
               "correlations": correlations(passages), "mismatches": mismatches(passages),
               "issues": by_issue(passages), "tone_vs_votes": tone_vs_votes(speeches)}
    for name, df in outputs.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
    plot_text_vs_voice(passages, OUT / "text_vs_voice.png")
    plot_issues(outputs["issues"], OUT / "tone_by_issue.png")
    plot_tone_vs_votes(speeches, OUT / "arousal_vs_dissent.png")

    log(f"\n{len(speeches)} speeches, {len(passages)} passages")
    log("speeches by language: " + ", ".join(f"{k} {v}" for k, v in
                                             speeches["language"].value_counts().items()))
    log("\nText sentiment vs voice, within speech:")
    log(outputs["correlations"].round(3).to_string(index=False))
    log("\nTone vs dissent on contested votes (descriptive):")
    log(outputs["tone_vs_votes"].round(3).to_string(index=False))
    log(f"\nwritten to {OUT}")
    return outputs


def analyze_cached():
    """Rerun the analysis on the cached per-speech results, without any models."""
    frames = [pd.read_csv(f) for f in sorted((OUT / "passages").glob("*.csv"))]
    passages = pd.concat([f for f in frames if len(f)], ignore_index=True)
    return analyze(passages)


def set_mode(languages):
    """Point OUT at the output folder for this mode ("all" or "en")."""
    global OUT
    OUT = OUT_EN if languages == "en" else OUT_ALL
    OUT.mkdir(parents=True, exist_ok=True)


def main(limit=None, whisper_model=None, device="cpu", redo=False, languages="all"):
    """`device` is used for all three models (Whisper falls back to cpu if it must)."""
    english_only = languages == "en"
    set_mode(languages)
    whisper_model = whisper_model or (WHISPER_MODEL_EN if english_only else WHISPER_MODEL)
    corpus = load_corpus()
    known = set(pd.concat([load_split("test")[0], load_split("test_heldout")[0]])["speech_id"])
    usable = corpus[corpus["speech_id"].isin(known) & corpus["audio_floor"].notna()]
    chosen = usable[usable["english"]] if english_only else usable
    print(f"{len(corpus)} speeches in the corpus, {corpus['english'].sum()} labelled English;"
          f" using {len(chosen)} ({'English only' if english_only else 'all languages'})"
          f" with audio and votes in our data")
    if limit:
        chosen = chosen.head(limit)
    print("loading models (first run downloads them)")
    passages = process(chosen, Whisper(whisper_model, device, english_only=english_only),
                       VoiceEmotion(device=device),
                       TextSentiment(TEXT_MODEL_EN if english_only else TEXT_MODEL, device=device),
                       redo=redo, english_only=english_only,
                       reuse=None if english_only else OUT_EN / "passages",
                       translate=None if english_only else Translator(device=device))
    return analyze(passages)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--languages", choices=["all", "en"], default="all",
                   help="all: every speech in its own language (default, results/audio_all/);"
                        " en: English-delivered speeches only (results/audio/)")
    p.add_argument("--limit", type=int, help="only the first N speeches")
    p.add_argument("--whisper-model", default=None,
                   help=f"faster-whisper model (default {WHISPER_MODEL}; {WHISPER_MODEL_EN} with"
                        " --languages en), e.g. medium")
    p.add_argument("--device", default="cpu", help="cpu or mps (Apple silicon) for all models")
    p.add_argument("--redo", action="store_true", help="recompute cached speeches")
    p.add_argument("--analyze-only", action="store_true",
                   help="skip audio processing; rerun the analysis on cached speeches")
    a = p.parse_args()
    if a.analyze_only:
        set_mode(a.languages)
        analyze_cached()
    else:
        main(a.limit, a.whisper_model, a.device, a.redo, a.languages)
