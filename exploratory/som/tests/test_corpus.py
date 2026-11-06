import numpy as np
import pandas as pd

from exploratory.som.corpus import clean_text
from exploratory.som.represent import ppmi, term_matrix, vocabulary
from exploratory.som.train import umatrix


def test_clean_text_removes_caption_markers_and_broken_timestamps():
    raw = "Na und [Musik] ist 00:00:27,830 --> 00:aN:NaN,NaN so >> schön [ __ ]"
    assert clean_text(raw) == "Na und ist so schön"


def test_vocabulary_merges_casing_and_needs_several_videos():
    df = pd.DataFrame({
        "video_id": ["a", "b", "c", "c"],
        "lemmas": ["Wahl Merz", "wahl Merz", "Wahl Rente", "Rente Rente"],
    })
    vocab = vocabulary(df, min_segments=2, min_videos=2, max_share=1.0)
    assert set(vocab.index) == {"wahl", "merz"}  # Rente: one video only
    assert vocab.at["wahl", "label"] == "Wahl"
    assert vocab.at["wahl", "segments"] == 3


def test_term_matrix_and_ppmi():
    df = pd.DataFrame({"lemmas": ["a b", "a a c"]})
    x = term_matrix(df, ["a", "b", "c"]).toarray()
    assert x.tolist() == [[1, 1, 0], [2, 0, 1]]
    weights = ppmi(np.array([[0, 5, 0], [5, 0, 1], [0, 1, 0]]))
    assert (weights >= 0).all() and weights[0, 2] == 0


def test_umatrix_is_zero_for_identical_prototypes():
    assert np.allclose(umatrix(np.ones((3, 4, 2))), 0)
