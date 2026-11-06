"""Clean the Apify youtube-scraper CSV export into one row per video.

Ported from ``transcript_cleaner.qmd``. On top of the plain-text
``transcript``, this keeps the caption cues with their start times so that
segments can later be linked back to a position in the video.
"""

import re

import pandas as pd

from .normalize import normalize_cues
from .segmentation import is_punctuated

# Flattened subtitle-text columns, e.g. "subtitles/0/srt", "subtitles/1/srt".
SRT_COLUMN_PATTERN = re.compile(r"^subtitles/(\d+)/srt$")
TRACK_FIELDS = ("type", "language")

# Timing line, e.g. "00:00:03,360 --> 00:00:07,200". The scraper sometimes
# drops zero padding ("00:00:3,360") and sometimes writes a broken end time
# ("00:00:27,830 --> 00:aN:NaN,NaN", about 1% of cues), so only the start
# time, which is all we use, has to be well-formed.
TIMING_PATTERN = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->.*")


def _seconds(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(value):
    """Parse a raw .srt string into a list of ``(start_seconds, text)`` cues.

    Empty cues (auto-generated captions alternate text cues with blank ones)
    and a cue repeating the previous cue's text are dropped.
    """
    if not isinstance(value, str):
        return []

    cues = []
    start = None
    text = []

    def flush():
        line = " ".join(text).strip()
        if line and (not cues or cues[-1][1] != line):
            cues.append((start, line))

    lines = value.splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        timing = TIMING_PATTERN.search(line)
        if timing:
            flush()
            start = _seconds(*timing.groups()[:4])
            text = []
            continue
        # Cue numbers precede a timing line; spoken numbers like "20" stay.
        if line.isdigit() and i + 1 < len(lines) and "-->" in lines[i + 1]:
            continue
        if line:
            text.append(line)
    flush()
    return cues


def clean_srt(value):
    """Convert a raw .srt string into plain text."""
    return " ".join(text for _, text in parse_srt(value))


def _first_track(row, srt_columns):
    """Index of the first caption track with cues, and its cues."""
    for col in srt_columns:
        cues = row[col]
        if cues:
            return int(SRT_COLUMN_PATTERN.match(col).group(1)), cues
    return None, []


def load_dataframe(path, normalize=True):
    """Load the scraper's CSV export into a cleaned, sorted DataFrame.

    Adds ``cues`` (list of ``(start_seconds, text)``) from the first
    non-empty caption track, ``caption_type`` and ``caption_language`` of
    that track (the Apify export marks auto-generated captions as
    ``auto_generated``), ``transcript`` (plain text after name
    normalisation, see ``normalize.py``), ``transcript_raw`` (before it),
    ``normalized`` (replacements made, e.g. ``"Habeck:3; Merz:1"``) and
    ``punctuated`` (at least one . ! ? per 100 words). Drops the raw SRT
    columns, renames ``id`` to ``video_id`` and sorts by ``type``.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc

    srt_columns = sorted(
        (c for c in df.columns if SRT_COLUMN_PATTERN.match(c)),
        key=lambda c: int(SRT_COLUMN_PATTERN.match(c).group(1)),
    )
    if srt_columns:
        parsed = df[srt_columns].apply(lambda col: col.map(parse_srt))
        picked = parsed.apply(lambda row: _first_track(row, srt_columns), axis=1)
        df["cues"] = picked.map(lambda t: t[1])
        track = [t[0] for t in picked]  # a list, so None stays None
        for field in TRACK_FIELDS:
            values = [
                df.at[i, f"subtitles/{n}/{field}"]
                if n is not None and f"subtitles/{n}/{field}" in df.columns
                else pd.NA
                for i, n in zip(df.index, track)
            ]
            df[f"caption_{field}"] = values
        df = df.drop(columns=srt_columns)
    else:
        df["cues"] = [[] for _ in range(len(df))]
        for field in TRACK_FIELDS:
            df[f"caption_{field}"] = pd.NA
    df["transcript_raw"] = df["cues"].map(lambda cues: " ".join(t for _, t in cues))
    if normalize:
        fixed = df["cues"].map(normalize_cues)
        df["cues"] = fixed.map(lambda t: t[0])
        df["normalized"] = fixed.map(
            lambda t: "; ".join(f"{k}:{v}" for k, v in sorted(t[1].items()))
        )
    else:
        df["normalized"] = ""
    df["transcript"] = df["cues"].map(lambda cues: " ".join(t for _, t in cues))
    # Most auto-generated captions before March 2025 have no sentence
    # punctuation, almost all from April 2025 on do (a change on YouTube's
    # side). Segmentation cuts unpunctuated captions at pauses instead.
    df["punctuated"] = df["transcript"].map(is_punctuated)

    if "type" not in df.columns:
        df["type"] = pd.Series(pd.NA, index=df.index, dtype="string")

    if "id" in df.columns:
        df = df.rename(columns={"id": "video_id"})

    if "date" in df.columns:
        # Most dates are full ISO timestamps, a few are date-only.
        df["date"] = pd.to_datetime(df["date"], format="ISO8601", utc=True)

    return df.sort_values(
        "type", kind="stable", na_position="last", ignore_index=True
    )
