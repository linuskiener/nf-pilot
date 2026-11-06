"""Per-video transcript quality columns for later analyses.

    caption_type, caption_language   from the scraper (see cleaning.py)
    punctuated                       sentence punctuation present
    n_words, words_per_minute        length and speech rate (duration needed)
    unknown_word_share               share of words (3+ letters) that are not
                                     in the wordfreq German lexicon; a proxy
                                     for recognition errors. New compounds
                                     ("Migrationswende") also count, so
                                     compare it between videos rather than
                                     reading it as an error rate.
    n_normalized                     name corrections made (normalize.py)
"""

import re

import pandas as pd

WORD = re.compile(r"[A-Za-zÄÖÜäöüß]{3,}")


def _duration_seconds(value):
    if pd.isna(value):
        return None
    parts = str(value).split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds or None


def unknown_word_share(text, cache={}):  # noqa: B006 - deliberate memo
    from wordfreq import zipf_frequency

    words = WORD.findall(str(text))
    if not words:
        return None
    unknown = 0
    for w in words:
        key = w.lower()
        if key not in cache:
            cache[key] = zipf_frequency(key, "de") == 0
        unknown += cache[key]
    return round(unknown / len(words), 4)


def quality_columns(videos):
    out = pd.DataFrame({"video_id": videos["video_id"]})
    out["n_words"] = videos["transcript"].str.split().str.len().fillna(0).astype(int)
    if "duration" in videos.columns:
        seconds = videos["duration"].map(_duration_seconds)
        out["words_per_minute"] = (out["n_words"] / (seconds / 60)).round(1)
    out["unknown_word_share"] = videos["transcript"].map(unknown_word_share)
    out["n_normalized"] = videos["normalized"].fillna("").map(
        lambda s: sum(int(p.rsplit(":", 1)[1]) for p in s.split("; ") if ":" in p
                      and "ambiguous" not in p)
    )
    return out
