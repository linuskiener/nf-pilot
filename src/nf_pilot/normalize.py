"""Conservative correction of speech-recognition errors in names.

YouTube's automatic captions spell many names phonetically ("Habbeck",
"Berbock", "Friedrich März"). That splits one actor into several strings,
which matters for counts, topic models and keyword-based relevance filters.

Two kinds of rules:

* ``resources/name_variants.csv``: plain regex replacements for strings that
  are not German words, so they can be replaced everywhere ("Habbeck" ->
  "Habeck"). Each row was checked against the 2025 H1 transcripts.
* Context rules in this module, for variants that ARE German words and may
  only be replaced when the context is clear:
  - "März" -> "Merz" only right after a person cue ("Friedrich", "Herr",
    "Kanzler", "CDU-Chef", ...). After a temporal cue ("im", "Anfang",
    "nächsten", a date, ...) it stays the month. Anything else stays as is
    and is counted as ambiguous, so the size of the grey zone is known.
  - "singen" -> "Singen" (the town) only after a place cue ("in", "nach",
    "aus", "bei", "km entfernt(en)").

Rules are applied to the caption cues before segmentation; the original
transcript is kept in ``transcript_raw``.
"""

import re
from collections import Counter
from pathlib import Path

import pandas as pd

VARIANTS_PATH = Path(__file__).parent / "resources" / "name_variants.csv"

PERSON_CUES = (
    r"Friedrich|Herr|Herrn|Herren|Kanzler|Bundeskanzler|Kanzlerkandidat|"
    r"Unionskanzlerkandidat|Kanzlerkandidaten|CDU-Chef|Unionschef|Parteichef|"
    r"CDU-Vorsitzende[nr]?|Fraktionschef|Oppositionsführer|Wahlsieger|"
    r"Friedrich-|Mister|Mr\.?|Regierung|Kanzlerschaft|Chef|Team|Koalition|"
    r"Unionsfraktionschef|Wahlkampf"
)
TEMPORAL_CUES = (
    r"im|Im|Anfang|anfang|Ende|ende|Mitte|mitte|seit|Seit|bis|Bis|ab|Ab|vom|"
    r"Vom|zum|am|Am|des|nächsten|letzten|kommenden|vergangenen|diesen|dieses|"
    r"Monat|Monats|bereits|schon|\d{1,2}\.?|ersten|zweiten|dritten"
)
MAERZ_PERSON = re.compile(rf"\b((?:{PERSON_CUES})\s+)[Mm]ärz\b")
MAERZ_TEMPORAL = re.compile(rf"\b(?:{TEMPORAL_CUES})\s+[Mm]ärz\b")
SINGEN_PLACE = re.compile(r"\b((?:in|nach|aus|bei|km entfernt(?:en)?)\s+)singen\b")


def load_variants(path=VARIANTS_PATH):
    table = pd.read_csv(path)
    return [(re.compile(p), r, kind) for p, r, kind in
            zip(table["pattern"], table["replacement"], table["kind"])]


VARIANTS = load_variants()


def normalize_text(text, variants=VARIANTS):
    """Return ``(text, counts)`` with all rules applied.

    ``counts`` holds the number of replacements per canonical form plus
    ``"März (ambiguous)"`` for occurrences of "März" left untouched without a
    temporal cue.
    """
    counts = Counter()
    for pattern, replacement, _ in variants:
        text, n = pattern.subn(replacement, text)
        if n:
            counts[re.sub(r"\\\d", "", replacement)] += n
    text, n = MAERZ_PERSON.subn(r"\1Merz", text)
    if n:
        counts["Merz"] += n
    text, n = SINGEN_PLACE.subn(r"\1Singen", text)
    if n:
        counts["Singen"] += n
    left = len(re.findall(r"\b[Mm]ärz\b", text))
    temporal = len(MAERZ_TEMPORAL.findall(text))
    if left - temporal > 0:
        counts["März (ambiguous)"] += left - temporal
    return text, counts


def normalize_cues(cues, variants=VARIANTS):
    """Normalise a list of ``(start, text)`` cues, keeping cue boundaries.

    Cues are joined with newlines so that a rule can see context across cue
    boundaries ("Friedrich" at the end of one cue, "März" at the start of the
    next); ``\\s`` in the patterns matches the newline.
    """
    if not cues:
        return cues, Counter()
    joined = "\n".join(text for _, text in cues)
    fixed, counts = normalize_text(joined, variants)
    parts = fixed.split("\n")
    if len(parts) != len(cues):  # a rule ate a newline; keep the original
        return cues, Counter()
    return [(start, part) for (start, _), part in zip(cues, parts)], counts
