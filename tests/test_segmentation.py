from src.nf_pilot.cleaning import clean_srt, parse_srt
from src.nf_pilot.segmentation import segment_video, split_sentences

SRT = """1
00:00:0,120 --> 00:00:3,360
Ganz Deutschland ist eine Baustelle.

2
00:00:1,750 --> 00:00:3,360
 

3
00:00:5,000 --> 00:00:8,000
Es kostet 20

4
00:00:8,500 --> 00:00:9,000
Euro. Wirklich!
"""


def test_parse_srt_drops_numbers_timings_and_blank_cues():
    assert parse_srt(SRT) == [
        (0.12, "Ganz Deutschland ist eine Baustelle."),
        (5.0, "Es kostet 20"),
        (8.5, "Euro. Wirklich!"),
    ]
    assert clean_srt(SRT) == (
        "Ganz Deutschland ist eine Baustelle. Es kostet 20 Euro. Wirklich!"
    )
    assert clean_srt(float("nan")) == ""


def test_split_sentences():
    sentences = [s for _, s in split_sentences("Eins. Zwei! „Drei?“ Vier")]
    assert sentences == ["Eins.", "Zwei!", "„Drei?“", "Vier"]


def test_segments_keep_cue_start_times():
    segments = segment_video(parse_srt(SRT), target_words=5, max_words=10)
    assert segments == [
        (0.12, "Ganz Deutschland ist eine Baustelle."),
        (5.0, "Es kostet 20 Euro. Wirklich!"),
    ]


def test_unpunctuated_cues_are_cut_at_the_longest_pause():
    cues = [
        (0.0, "eins zwei drei"),
        (1.0, "vier fünf sechs"),
        (5.0, "sieben acht neun"),  # long pause before the next cue
        (12.0, "zehn elf zwölf"),
        (13.0, "dreizehn vierzehn"),
    ]
    segments = segment_video(cues, target_words=4, max_words=12)
    assert segments == [
        (0.0, "eins zwei drei vier fünf sechs sieben acht neun"),
        (12.0, "zehn elf zwölf dreizehn vierzehn"),
    ]


def test_music_marker_is_a_break():
    cues = [(0.0, "a b c"), (1.0, "d e f"), (2.0, "[Musik]"), (3.0, "g h i")]
    segments = segment_video(cues, target_words=2, max_words=12)
    assert segments[0] == (0.0, "a b c d e f")
