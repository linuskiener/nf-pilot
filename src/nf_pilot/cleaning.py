"""Clean the Apify youtube-scraper CSV export into one row per video.

Ported from ``transcript_cleaner.qmd``. On top of the plain-text
``transcript``, this keeps the caption cues with their start times so that
segments can later be linked back to a position in the video.
"""

import re

import pandas as pd

# Flattened subtitle-text columns, e.g. "subtitles/0/srt", "subtitles/1/srt".
SRT_COLUMN_PATTERN = re.compile(r"^subtitles/(\d+)/srt$")

# Timing line, e.g. "00:00:03,360 --> 00:00:07,200". The scraper sometimes
# drops zero padding ("00:00:3,360"), so every field is matched loosely.
TIMING_PATTERN = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)"
)


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


def load_dataframe(path):
    """Load the scraper's CSV export into a cleaned, sorted DataFrame.

    Adds ``transcript`` (plain text), ``cues`` (list of
    ``(start_seconds, text)``) from the first non-empty caption track and
    ``punctuated`` (at least one . ! ? per 100 words), drops
    the raw SRT columns, renames ``id`` to ``video_id`` and sorts by ``type``.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc

    srt_columns = sorted(c for c in df.columns if SRT_COLUMN_PATTERN.match(c))
    if srt_columns:
        parsed = df[srt_columns].apply(lambda col: col.map(parse_srt))
        df["cues"] = parsed.apply(
            lambda row: next((cues for cues in row if cues), []), axis=1
        )
        df = df.drop(columns=srt_columns)
    else:
        df["cues"] = [[] for _ in range(len(df))]
    df["transcript"] = df["cues"].map(lambda cues: " ".join(t for _, t in cues))
    # About half of the auto-generated captions come without any sentence
    # punctuation; those can only be segmented by length until punctuation
    # is restored.
    words = df["transcript"].str.split().str.len()
    df["punctuated"] = (words > 0) & (
        df["transcript"].str.count(r"[.!?]") >= 0.01 * words
    )

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
