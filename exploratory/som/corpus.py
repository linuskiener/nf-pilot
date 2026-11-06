"""Turn segmented transcripts into lists of content lemmas.

Input is the per-segment table that ``src.nf_pilot.prepare_corpus`` writes
(``*_SEG.csv``, about 200 words per segment). This module only reads that
output; it does not change the frame-detection pipeline.

Why lemmas of content words: a SOM over bag-of-words vectors never sees word
order or sentence boundaries, so the missing punctuation in about half of the
auto-generated captions does not matter here. What does matter is German
inflection ("Wahl", "Wahlen", "Wahlkampfes") and the filler words of spoken
language ("ja", "halt", "mal"), so both are normalised away.
"""

import re
from pathlib import Path

import pandas as pd

# Caption markers such as [Musik], [Applaus], [Gelächter] and the censored
# profanity placeholder "[ __ ]", plus speaker-change arrows.
# Also SRT timing lines the scraper broke ("00:00:27,830 --> 00:aN:NaN,NaN"),
# which the cleaner in src/nf_pilot does not recognise and leaves in the text
# of about 1 in 9 segments.
CAPTION_MARKERS = re.compile(
    r"\[[^\]]{0,40}\]|>>+|&\w+;|[\d:,.]*\s*-->\s*[\w:,.]*"
)

KEEP_POS = {"NOUN", "PROPN", "ADJ", "VERB"}

# Spoken-language fillers and light verbs that survive the spaCy stop list but
# carry no topic. Kept short on purpose: PPMI/TF-IDF already down-weight
# frequent words, this only removes the worst offenders.
EXTRA_STOPWORDS = {
    "ja", "halt", "mal", "eben", "genau", "okay", "ok", "naja", "äh", "ähm",
    "hm", "na", "irgendwie", "quasi", "sozusagen", "eigentlich", "wirklich",
    "einfach", "natürlich", "übrigens", "letztlich", "jetzt", "heute",
    "sagen", "machen", "geben", "gehen", "kommen", "sehen", "wissen", "finden",
    "lassen", "stehen", "bleiben", "glauben", "denken", "meinen", "heißen",
    "bringen", "nehmen", "halten", "stellen", "sprechen", "reden", "fragen",
    "brauchen", "kriegen", "bekommen", "zeigen", "nennen", "tun", "schauen",
    "gut", "groß", "klein", "neu", "alt", "ganz", "viel", "richtig", "erst",
    "letzt", "nächst", "weit", "lang", "hoch", "spannend", "interessant",
    "ding", "sache", "mensch", "leute", "frage", "jahr", "zeit", "tag", "mal",
    "prozent", "teil", "art", "weise", "fall", "punkt", "thema", "beispiel",
    "dank", "danke", "bitte", "hallo", "herzlich", "willkommen",
    "letzter", "nächster", "weiterer", "bisschen", "gerade", "wichtig", "klar",
    "schön", "bestimmt", "verschieden", "eigen", "passieren", "versuchen",
    "gehören", "hören", "erleben", "schreiben", "führen", "stelle", "seite",
    "grund", "wort", "woche", "problem", "anderer", "herr", "einzig",
}


def clean_text(text):
    """Remove caption markers and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", CAPTION_MARKERS.sub(" ", text)).strip()


def load_segments(paths):
    """Read one or more ``*_SEG.csv`` files into one DataFrame.

    Adds ``source`` (the run name, e.g. ``niusde_2025_H1``) and ``month``.
    """
    frames = []
    for path in map(Path, paths):
        df = pd.read_csv(path)
        df["source"] = path.name.removesuffix("_SEG.csv")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["clean"] = df["text"].map(clean_text)
    return df


def load_nlp():
    import spacy

    # The morphologizer sets the coarse POS used for filtering; the parser
    # and NER are not needed and would triple the runtime.
    return spacy.load("de_core_news_sm", disable=["parser", "ner"])


def content_lemma(token, stopwords):
    """The token's lemma if it is a content word, else ``None``.

    Trailing punctuation is stripped first, because in unpunctuated captions
    spaCy sometimes glues a full stop to a name ("AfD."); hyphenated
    compounds such as "AfD-Verbot" are kept whole.
    """
    if token.pos_ not in KEEP_POS:
        return None
    lemma = token.lemma_.strip(" .,;:!?\"'()")
    if (
        len(lemma) > 2
        and lemma.replace("-", "").isalpha()
        and lemma.lower() not in stopwords
    ):
        return lemma
    return None


def lemmatize(texts, nlp=None, n_process=1, batch_size=64):
    """Return one list of content lemmas per text.

    Lemmas keep their casing (German nouns are capitalised, which makes the
    map easier to read); casing variants are merged later in ``vocabulary``.
    """
    nlp = nlp or load_nlp()
    stopwords = {w.lower() for w in nlp.Defaults.stop_words} | EXTRA_STOPWORDS
    out = []
    for doc in nlp.pipe(texts, n_process=n_process, batch_size=batch_size):
        lemmas = (content_lemma(t, stopwords) for t in doc)
        out.append([lemma for lemma in lemmas if lemma])
    return out


def lemmatize_segments(df, cache=None, n_process=1):
    """Add a ``lemmas`` column (space-joined), cached as a CSV by segment id."""
    if cache is not None and Path(cache).exists():
        cached = pd.read_csv(cache, keep_default_na=False).set_index("segment_id")
        if df["segment_id"].isin(cached.index).all():
            df["lemmas"] = df["segment_id"].map(cached["lemmas"])
            return df
    lemmas = lemmatize(df["clean"].tolist(), n_process=n_process)
    df["lemmas"] = [" ".join(seq) for seq in lemmas]
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        df[["segment_id", "lemmas"]].to_csv(cache, index=False)
    return df
