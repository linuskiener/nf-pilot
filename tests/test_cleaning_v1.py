import pandas as pd

from src.nf_pilot.boilerplate import segment_role
from src.nf_pilot.cleaning import parse_srt
from src.nf_pilot.derivation import confidence, find_sources
from src.nf_pilot.normalize import normalize_cues, normalize_text


def test_broken_end_time_is_still_a_timing_line():
    srt = "1\n00:00:27,830 --> 00:aN:NaN,NaN\nSo schön.\n\n2\n00:00:30,000 --> 00:00:31,000\nJa."
    assert parse_srt(srt) == [(27.83, "So schön."), (30.0, "Ja.")]


def test_maerz_is_merz_only_after_a_person_cue():
    text, counts = normalize_text(
        "Friedrich März sagt, Kanzler März will im März und Anfang März. März hat recht."
    )
    assert text == ("Friedrich Merz sagt, Kanzler Merz will im März und Anfang März. "
                    "März hat recht.")
    assert counts["Merz"] == 2
    assert counts["März (ambiguous)"] == 1
    assert normalize_text("am 3. März")[1]["März (ambiguous)"] == 0


def test_name_variants_and_their_exceptions():
    text, _ = normalize_text(
        "habbeck und Berbock in erschaffenburg, Elon mask, Till Lindemann, "
        "Lindemann von der CDU, in singen wird gesungen, wir singen, die Maske"
    )
    assert text == (
        "Habeck und Baerbock in Aschaffenburg, Elon Musk, Till Lindemann, "
        "Linnemann von der CDU, in Singen wird gesungen, wir singen, die Maske"
    )


def test_normalize_cues_sees_across_cue_boundaries():
    cues, counts = normalize_cues([(0.0, "das sagt Friedrich"), (1.5, "März heute")])
    assert cues == [(0.0, "das sagt Friedrich"), (1.5, "Merz heute")]
    assert counts["Merz"] == 1


def test_clip_is_found_in_its_source_with_dates():
    from itertools import product

    words = ["".join(p) for p in product("abcdefg", repeat=3)][:300]
    speech = " ".join(words)
    videos = pd.DataFrame({
        "video_id": ["stream", "short", "other"],
        "type": ["stream", "shorts", "shorts"],
        "date": ["2025-03-01T10:00Z", "2025-03-01T18:00Z", "2025-03-02T10:00Z"],
        "transcript": [speech + " ende der sendung",
                       " ".join(speech.split()[100:160]),
                       "ganz anderer text über etwas völlig anderes heute"],
    })
    out = find_sources(videos).set_index("video_id")
    assert out.at["short", "derived_from"] == "stream"
    assert out.at["short", "derived_confidence"] == "high"
    assert abs(out.at["short", "derived_lag_days"] - 1 / 3) < 0.01
    assert out.at["other", "derived_confidence"] == "none"
    assert confidence(0.9, -5) == "medium"  # source published long after


def test_segment_roles():
    assert segment_role("Abonnieren Sie unseren Kanal und liken Sie das Video") == "outro"
    assert segment_role("Zum Wetter: sonnig bei 20 Grad") == "weather"
    assert segment_role("Die Regierung plant eine Reform der Rente.") == "content"
