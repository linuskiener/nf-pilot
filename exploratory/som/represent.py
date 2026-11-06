"""Vector representations for the two maps.

Word map (the "galaxy of concepts" in Play Among Books): every lemma is one
observation. In the book a word is described by its frequencies across the
229 books of the library; here the "books" are the ~200-word transcript
segments, which are many more and all about the same length. A word is
therefore described by which other words it shares segments with, weighted
by positive pointwise mutual information (PPMI) and compressed with a
truncated SVD. Words used in the same stretches of talk end up close.

Segment map (WEBSOM / Bungum & Gambäck): every segment is one observation,
described by its TF-IDF vector over the same vocabulary, again compressed by
SVD. This is the map that metadata (month, format, views) and, later, frame
labels can be laid over.

Both representations are L2-normalised, so the Euclidean distance the SOM
uses ranks neighbours the same way cosine similarity does.
"""

from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .corpus import EXTRA_STOPWORDS


def vocabulary(df, min_segments=20, min_videos=8, max_share=0.25):
    """Choose terms and their display form.

    A term must occur in at least ``min_segments`` segments and in at least
    ``min_videos`` different videos (so one long livestream cannot put its
    own transcription errors on the map), and in at most ``max_share`` of all
    segments. Casing variants are merged; the most frequent one is displayed.

    Returns a DataFrame indexed by the lower-cased term with ``label``,
    ``segments``, ``videos`` and ``tokens``, sorted by ``segments``.
    """
    surface = Counter()
    seg_sets = {}
    vid_sets = {}
    tokens = Counter()
    for seg_idx, (video, lemmas) in enumerate(zip(df["video_id"], df["lemmas"])):
        for lemma in lemmas.split():
            key = lemma.lower()
            surface[(key, lemma)] += 1
            tokens[key] += 1
            seg_sets.setdefault(key, set()).add(seg_idx)
            vid_sets.setdefault(key, set()).add(video)
    rows = []
    for key, segs in seg_sets.items():
        rows.append((key, len(segs), len(vid_sets[key]), tokens[key]))
    vocab = pd.DataFrame(rows, columns=["term", "segments", "videos", "tokens"])
    vocab = vocab[
        (vocab["segments"] >= min_segments)
        & (vocab["videos"] >= min_videos)
        & (vocab["segments"] <= max_share * len(df))
        # Re-applied here so a stop-list change works on cached lemmas too.
        & ~vocab["term"].isin(EXTRA_STOPWORDS)
    ]
    best = {}
    for (key, form), n in surface.items():
        if n > best.get(key, ("", 0))[1]:
            best[key] = (form, n)
    vocab["label"] = vocab["term"].map(lambda k: best[k][0])
    return vocab.sort_values("segments", ascending=False).set_index("term")


def term_matrix(df, terms, binary=False):
    """Sparse segment x term count matrix over ``terms`` (lower-cased)."""
    index = {t: i for i, t in enumerate(terms)}
    rows, cols, vals = [], [], []
    for r, lemmas in enumerate(df["lemmas"]):
        counts = Counter(index[w] for w in lemmas.lower().split() if w in index)
        for c, n in counts.items():
            rows.append(r)
            cols.append(c)
            vals.append(1.0 if binary else float(n))
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(len(df), len(terms)), dtype=np.float32
    )


def ppmi(counts, alpha=0.75):
    """Positive PMI of a dense co-occurrence matrix, with context smoothing."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    row = counts.sum(axis=1, keepdims=True) / total
    ctx = counts.sum(axis=0) ** alpha
    ctx = (ctx / ctx.sum())[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((counts / total) / (row * ctx))
    pmi[~np.isfinite(pmi)] = 0.0
    return np.maximum(pmi, 0.0)


def word_vectors(df, vocab, n_words=3000, dims=200, seed=42):
    """PPMI-SVD vectors for the ``n_words`` most widespread terms.

    Contexts are all vocabulary terms; co-occurrence means sharing a segment.
    Returns ``(vectors, words)`` where ``words`` is the vocab slice in order.
    """
    words = vocab.head(n_words)
    x = term_matrix(df, vocab.index, binary=True)
    co = (x[:, : len(words)].T @ x).toarray()
    np.fill_diagonal(co[:, : len(words)], 0.0)
    weights = ppmi(co)
    svd = TruncatedSVD(n_components=dims, random_state=seed)
    vectors = svd.fit_transform(weights)
    return normalize(vectors).astype(np.float32), words


def segment_vectors(df, vocab, dims=200, seed=42):
    """TF-IDF-SVD vectors for every segment.

    Returns ``(vectors, keep, svd, idf)``: ``keep`` masks the segments that
    have at least three vocabulary terms (the rest cannot be placed sensibly),
    ``svd.components_`` maps prototypes back to term weights for labelling.
    """
    counts = term_matrix(df, vocab.index)
    keep = np.asarray((counts > 0).sum(axis=1)).ravel() >= 3
    counts = counts[keep]
    tf = counts.copy()
    tf.data = 1.0 + np.log(tf.data)
    df_count = np.asarray((counts > 0).sum(axis=0)).ravel()
    idf = np.log((1 + counts.shape[0]) / (1 + df_count)) + 1.0
    tfidf = normalize(tf @ sparse.diags(idf.astype(np.float32)))
    svd = TruncatedSVD(n_components=dims, random_state=seed)
    vectors = svd.fit_transform(tfidf)
    return normalize(vectors).astype(np.float32), keep, svd, tfidf
