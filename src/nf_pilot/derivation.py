"""Find Shorts and videos that re-use speech from a longer upload.

Many Shorts are cut from livestreams or long videos, so the same speech is
transcribed twice. Downstream analyses compare formats, so nothing is
dropped; instead each video and segment gets a pointer to its most likely
source and a confidence.

Method: every transcript becomes a set of word 4-grams (lower-cased letters
only, so punctuation and capitalisation differences between caption versions
do not matter). For each video, the *containment* of its 4-grams in each
longer upload is measured: the share of its own 4-grams that also occur
there. Re-used speech gives high containment even when the clip is a tiny
part of the source; two different videos on the same topic share well under
5% of their 4-grams.

Publication dates are the cross-check: a clip is normally published after
its source. ``derived_lag_days`` = clip date - source date, so it is
positive for a clip published after its source.

Confidence
    high    containment >= 0.5 and the source was published at most one day
            after the clip (allows premieres and time zones)
    medium  containment >= 0.5 but the source is younger, or
            0.25 <= containment < 0.5 with plausible dates
    low     0.1 <= containment < 0.25
    none    below 0.1
"""

import re
from collections import defaultdict

import numpy as np
import pandas as pd

N = 4
WORD = re.compile(r"[a-zäöüß]+")


def shingles(text, n=N):
    words = WORD.findall(str(text).lower())
    return {hash(tuple(words[i:i + n])) for i in range(len(words) - n + 1)}


def confidence(containment, lag_days):
    if containment >= 0.5:
        return "high" if lag_days >= -1 else "medium"
    if containment >= 0.25:
        return "medium" if lag_days >= -1 else "low"
    if containment >= 0.1:
        return "low"
    return "none"


def find_sources(videos, min_shared=5):
    """Return one row per video with its most likely source.

    ``videos`` needs ``video_id``, ``type``, ``date`` and ``transcript``.
    Only uploads with more words than the candidate are considered sources.
    """
    sets = [shingles(t) for t in videos["transcript"]]
    sizes = np.array([len(s) for s in sets])
    index = defaultdict(list)
    for j, s in enumerate(sets):
        for h in s:
            index[h].append(j)
    ids = videos["video_id"].to_numpy()
    dates = pd.to_datetime(videos["date"], utc=True).to_numpy()
    rows = []
    for i, s in enumerate(sets):
        best = (None, 0.0)
        if s:
            hits = defaultdict(int)
            for h in s:
                for j in index[h]:
                    if j != i and sizes[j] > sizes[i]:
                        hits[j] += 1
            for j, k in hits.items():
                if k >= min_shared and k / len(s) > best[1]:
                    best = (j, k / len(s))
        j, c = best
        lag = (dates[i] - dates[j]) / np.timedelta64(1, "D") if j is not None else np.nan
        rows.append({
            "video_id": ids[i],
            "derived_from": ids[j] if j is not None else pd.NA,
            "derived_from_type": videos["type"].iloc[j] if j is not None else pd.NA,
            "derived_containment": round(c, 3),
            "derived_lag_days": round(lag, 2) if j is not None else np.nan,
            "derived_confidence": confidence(c, lag) if j is not None else "none",
        })
    return pd.DataFrame(rows)


def segment_containment(segments, videos, sources):
    """Containment of every segment of a derived video in its source.

    Adds ``duplicate_of`` (source video id) and ``duplicate_containment`` to
    ``segments``; ``duplicate_of`` is set where the segment itself is at
    least half contained and its video's confidence is high or medium.
    """
    by_id = dict(zip(videos["video_id"], videos["transcript"]))
    src = sources.set_index("video_id")
    cache = {}
    contained, dup_of = [], []
    for vid, text in zip(segments["video_id"], segments["text"]):
        row = src.loc[vid] if vid in src.index else None
        if row is None or row["derived_confidence"] in ("none", "low"):
            contained.append(np.nan)
            dup_of.append(pd.NA)
            continue
        source = row["derived_from"]
        if source not in cache:
            cache[source] = shingles(by_id[source])
        s = shingles(text)
        c = len(s & cache[source]) / len(s) if s else 0.0
        contained.append(round(c, 3))
        dup_of.append(source if c >= 0.5 else pd.NA)
    segments = segments.copy()
    segments["duplicate_containment"] = contained
    segments["duplicate_of"] = dup_of
    return segments
