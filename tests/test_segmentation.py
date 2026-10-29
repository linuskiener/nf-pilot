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
