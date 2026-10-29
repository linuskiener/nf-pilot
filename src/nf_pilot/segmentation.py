"""Split cleaned transcripts into paragraph-sized segments.

v0 is length-based: sentences are packed greedily until a segment reaches
``target_words``. Segment boundaries always fall between sentences, and each
segment keeps the start time of the caption cue it begins in. Topic-aware
boundaries (e.g. embedding similarity between neighbouring sentences) can
replace ``pack_sentences`` later without changing the output format.
"""

import bisect
import re

import pandas as pd

# A sentence ends with . ! ? (optionally followed by a closing quote) and is
# followed by whitespace and an upper-case letter, digit or opening quote.
SENTENCE_END = re.compile(
    r"(?:(?<=[.!?])|(?<=[.!?][\"“”»]))\s+(?=[\"„“«A-ZÄÖÜ0-9])"
)

# Caption markers for music, applause and laughter.
MARKER = re.compile(r"\[(musik|applaus|gelächter)\]", re.IGNORECASE)


def split_sentences(text):
    """Split text into ``(char_offset, sentence)`` pairs."""
    sentences = []
    pos = 0
    for match in SENTENCE_END.finditer(text):
        sentence = text[pos:match.start()].strip()
        if sentence:
            sentences.append((pos, sentence))
        pos = match.end()
    if text[pos:].strip():
        sentences.append((pos, text[pos:].strip()))
    return sentences


def pack_sentences(sentences, target_words=200, max_words=300):
    """Group ``(offset, sentence)`` pairs into segments of about target_words.

    A segment is closed once it reaches ``target_words``, or before a
    sentence that would push it past ``max_words``. A single sentence longer
    than ``max_words`` (common in unpunctuated stretches of auto-captions) is
    cut into ``max_words`` chunks.
    """
    segments = []
    current, count = [], 0

    def close():
        nonlocal current, count
        if current:
            segments.append((current[0][0], " ".join(s for _, s in current)))
        current, count = [], 0

    for offset, sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            close()
            for i in range(0, len(words), max_words):
                segments.append((offset, " ".join(words[i:i + max_words])))
            continue
        if count and count + len(words) > max_words:
            close()
        current.append((offset, sentence))
        count += len(words)
        if count >= target_words:
            close()
    close()
    return segments


def _break_score(cues, i):
    """How likely a segment boundary is after cue ``i``.

    Auto-captions overlap, so cue end times carry no pause information. The
    time until the next cue starts, per word of cue ``i``, does: on the
    punctuated captions, the slowest 10% of cues end a sentence about three
    times as often as the average cue. Music, applause or laughter markers
    are treated as certain breaks.
    """
    start, text = cues[i]
    if i + 1 >= len(cues):
        return float("inf")
    next_start, next_text = cues[i + 1]
    if MARKER.search(text) or MARKER.search(next_text):
        return float("inf")
    return (next_start - start) / max(len(text.split()), 1)


def pack_cues(cues, target_words=200, max_words=300):
    """Group unpunctuated cues into segments, cutting at the likeliest pause.

    Each segment ends at the cue with the highest break score among those
    that keep it between ``0.75 * target_words`` and ``max_words`` words.
    """
    min_words = int(0.75 * target_words)
    counts = [len(text.split()) for _, text in cues]
    segments = []
    a = 0
    while a < len(cues):
        best, best_score, words = None, -1.0, 0
        for j in range(a, len(cues)):
            words += counts[j]
            if words > max_words and best is not None:
                break
            if words >= min_words or j == len(cues) - 1:
                score = _break_score(cues, j)
                if score > best_score:
                    best, best_score = j, score
            if words > max_words:
                break
        if best is None:
            best = j
        text = " ".join(t for _, t in cues[a:best + 1])
        segments.append((cues[a][0], text))
        a = best + 1
    return segments


def is_punctuated(text):
    """At least one sentence-final . ! ? per 100 words."""
    words = len(text.split())
    return words > 0 and len(re.findall(r"[.!?]", text)) >= 0.01 * words


def segment_video(cues, target_words=200, max_words=300):
    """Segment one video's cues into ``(start_seconds, text)`` pairs.

    Punctuated captions are cut at sentence boundaries, unpunctuated ones at
    the likeliest pauses between cues.
    """
    if not cues:
        return []
    if not is_punctuated(" ".join(text for _, text in cues)):
        return pack_cues(cues, target_words, max_words)
    # Rebuild the transcript exactly as cleaning.load_dataframe does and
    # remember where each cue starts, to map segments back to video time.
    offsets, starts, pos = [], [], 0
    for start, text in cues:
        offsets.append(pos)
        starts.append(start)
        pos += len(text) + 1
    transcript = " ".join(text for _, text in cues)

    segments = pack_sentences(
        split_sentences(transcript), target_words, max_words
    )
    return [
        (starts[bisect.bisect_right(offsets, offset) - 1], text)
        for offset, text in segments
    ]


def segment_dataframe(df, target_words=200, max_words=300):
    """Turn a cleaned video DataFrame into one row per segment."""
    rows = []
    for video in df.itertuples(index=False):
        segments = segment_video(video.cues, target_words, max_words)
        for n, (start, text) in enumerate(segments):
            rows.append({
                "segment_id": f"{video.video_id}_{n:04d}",
                "video_id": video.video_id,
                "channel": getattr(video, "channelName", None),
                "type": video.type,
                "date": video.date,
                "title": video.title,
                "punctuated": video.punctuated,
                "position": n,
                "n_segments": len(segments),
                "start_sec": int(start),
                "url": f"https://www.youtube.com/watch?v={video.video_id}&t={int(start)}s",
                "n_words": len(text.split()),
                "text": text,
            })
    return pd.DataFrame(rows)
