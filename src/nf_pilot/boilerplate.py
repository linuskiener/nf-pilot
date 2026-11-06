"""Flag segments that are show furniture rather than content.

Heuristic v0, built from the segment-map regions that collected these
segments. Each role has a list of phrases; a segment gets the role whose
phrases it matches most, counting each phrase once, if at least
``MIN_HITS`` different phrases match, otherwise ``content``. Flags only: nothing is removed, so every analysis can decide.

    intro         greetings and show openings
    outro         sign-offs, "subscribe / like / comment", app and website plugs
    press_review  reading out other outlets' headlines
    weather       weather forecast at the end of news bulletins
"""

import re

MIN_HITS = 2

ROLES = {
    "intro": [
        r"herzlich willkommen", r"guten morgen", r"guten abend",
        r"hier ist nius", r"der tag beginnt mit nius", r"schön, dass (?:du|sie|ihr) da",
        r"willkommen (?:zu|zur|bei|im)", r"ich begrüße",
        r"darüber spricht deutschland", r"es ist (?:montag|dienstag|mittwoch|donnerstag|freitag|samstag|sonntag), der",
    ],
    "outro": [
        r"abonnier", r"(?:unseren|unserem|youtube-?)\s?kanal", r"\blike\b", r"liken",
        r"kommentieren sie", r"weiterempfehl",
        r"sagen sie es (?:bitte )?weiter", r"glocke", r"\bapp\b", r"nius\.de",
        r"bis morgen", r"bis zum nächsten mal", r"tschüss", r"auf wiedersehen",
        r"vielen dank fürs (?:zuschauen|zuhören)", r"danke fürs", r"shop\.nius",
        r"nur selber denken macht schlau", r"folgen sie uns",
    ],
    "press_review": [
        r"presseschau", r"titelt", r"titelseite", r"schlagzeile", r"aufmacher",
        r"schreibt (?:die|der|das) (?:bild|welt|faz|süddeutsche|taz|spiegel|zeit|focus|nzz|tagesspiegel)",
        r"kommentiert (?:die|der) ", r"in (?:der|die) (?:bild|welt|faz|taz|süddeutschen|zeit)",
    ],
    "weather": [
        r"zum wetter", r"vorhersage", r"\d+\s*(?:grad|°)", r"\bregen\b", r"sonnig",
        r"bewölkt", r"gewitter", r"wetterdienst", r"temperaturen", r"wolken",
    ],
}
PATTERNS = {role: [re.compile(p) for p in pats] for role, pats in ROLES.items()}


def role_hits(text):
    low = str(text).lower()
    return {role: sum(1 for p in pats if p.search(low))
            for role, pats in PATTERNS.items()}


def segment_role(text, min_hits=MIN_HITS):
    hits = role_hits(text)
    role, n = max(hits.items(), key=lambda kv: kv[1])
    return role if n >= min_hits else "content"


def flag_segments(segments):
    segments = segments.copy()
    segments["segment_role"] = segments["text"].map(segment_role)
    return segments
