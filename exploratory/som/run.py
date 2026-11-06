"""Train the exploratory word map and segment map and write all figures.

Usage (from the repo root, after ``prepare_corpus`` has written the segments):

    python -m exploratory.som.run data/processed/niusde_2025_H1_SEG.csv \
        --videos data/raw/niusde_2025_H1_DATA.csv

Several ``*_SEG.csv`` files (e.g. several half-years) can be passed at once;
they are mapped together so the half-years can be compared on one map.
Everything is written to ``exploratory/som/output/<name>/`` (not committed).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .corpus import lemmatize_segments, load_segments
from .represent import segment_vectors, term_matrix, vocabulary, word_vectors
from .train import train_som

DEFAULT_TERMS = [
    "migration", "merz", "afd", "brandmauer", "trump", "ukraine", "messer",
    "klima", "musk", "israel", "meinungsfreiheit", "bürgergeld",
]


def log_ratio(part, whole, part_total, whole_total, prior=10.0):
    """log2 of observed over expected count of a cell in one slice.

    ``prior`` pseudo-counts are added to both, which shrinks cells with
    little data towards 0 (white) so that only well-supported contrasts
    show up in colour.
    """
    expected = whole * part_total / whole_total
    return np.log2((part + prior) / (expected + prior))


def cells_of(trained, labels, order=None):
    """Map (row, col) -> labels of the samples whose BMU it is."""
    cells = {}
    idx = range(len(labels)) if order is None else order
    for i in idx:
        r, c = trained.bmu[i]
        cells.setdefault((int(r), int(c)), []).append(labels[i])
    return cells


def region_centres(regions):
    out = {}
    for k in np.unique(regions):
        rr, cc = np.nonzero(regions == k)
        # The member cell closest to the centroid, so the label sits inside.
        i = np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)
        out[int(k)] = (int(rr[i]), int(cc[i]))
    return out


# --------------------------------------------------------------------------
# Word map
# --------------------------------------------------------------------------


def run_word_map(df, vocab, out, args):
    vectors, words = word_vectors(df, vocab, n_words=args.words, dims=args.dims)
    m = train_som(vectors, args.word_rows, args.word_cols, epochs=args.epochs,
                  n_regions=args.word_regions)
    labels = words["label"].tolist()
    tokens = words["tokens"].to_numpy()
    cells = cells_of(m, labels, order=np.argsort(-tokens))  # frequent first

    regions = m.clusters
    region_rows = []
    for k in np.unique(regions):
        members = [i for i in np.argsort(-tokens) if regions[tuple(m.bmu[i])] == k]
        region_rows.append({
            "region": int(k),
            "cells": int((regions == k).sum()),
            "words": len(members),
            "top_words": ", ".join(labels[i] for i in members[:12]),
        })
    region_table = pd.DataFrame(region_rows)
    region_table.to_csv(out / "word_regions.csv", index=False)
    pd.DataFrame({
        "word": labels, "row": m.bmu[:, 0], "col": m.bmu[:, 1],
        "region": [int(regions[tuple(b)]) for b in m.bmu],
        "segments": words["segments"].to_numpy(), "tokens": tokens,
    }).to_csv(out / "word_cells.csv", index=False)

    plots.word_grid(out / "01_word_grid.png", cells, m.rows, m.cols, m.umatrix,
                    regions, f"Word map: {len(labels)} lemmas on a "
                    f"{m.rows}x{m.cols} grid ({args.name})")
    plots.word_grid(out / "01_word_grid.pdf", cells, m.rows, m.cols, m.umatrix,
                    regions, f"Word map ({args.name})")
    plots.single(out / "02_word_umatrix.png", m.umatrix,
                 "Word map, U-matrix: dark = border between word fields",
                 "mean distance to neighbours", cmap=plots.GREY, regions=regions)
    hits = m.hits.astype(float)
    hits[hits == 0] = np.nan
    plots.single(out / "03_word_hits.png", hits,
                 "Word map, hit map: words per cell (white = empty cell)",
                 "words", regions=regions)
    plots.training_curves(out / "04_word_training.png", m.qe, m.te,
                          "Word map training")

    # Spectral faces (Play Among Books): how strongly each cell's words are
    # used in one slice of the corpus compared to the corpus as a whole.
    counts = term_matrix(df, words.index)  # segment x word token counts
    node_of_word = m.node
    n_nodes = m.rows * m.cols

    def node_totals(mask):
        per_word = np.asarray(counts[mask].sum(axis=0)).ravel()
        totals = np.zeros(n_nodes)
        np.add.at(totals, node_of_word, per_word)
        return totals

    whole = node_totals(np.ones(len(df), bool))
    faces = {}
    for column, min_size in ((args.period, 500), ("type", 0)):
        maps, titles = [], []
        for value, group in df.groupby(column):
            if len(group) < min_size:
                continue
            part = node_totals(df[column].eq(value).to_numpy())
            face = log_ratio(part, whole, part.sum(), whole.sum(), prior=30)
            face[whole == 0] = np.nan
            maps.append(face.reshape(m.rows, m.cols))
            titles.append(f"{value}  ({len(group)} segments)")
            faces[f"{column}:{value}"] = face
        plots.small_multiples(
            out / f"05_word_faces_{'period' if column == args.period else column}.png",
            maps, titles,
            f"Word map faces by {column}: red = this cell's words are used "
            "more than usual, blue = less", regions=regions,
            cbar_label="log2 share vs. whole corpus")

    summary = {
        "words": len(labels), "rows": m.rows, "cols": m.cols,
        "quantisation_error": round(m.final_qe, 4),
        "topographic_error_pct": round(m.final_te, 2),
        "empty_cells": int((m.hits == 0).sum()),
    }
    export = {
        "rows": m.rows, "cols": m.cols,
        "cells": [[cells.get((r, c), []) for c in range(m.cols)]
                  for r in range(m.rows)],
        "regions": regions.tolist(), "umatrix": np.round(m.umatrix, 4).tolist(),
        "faces": {k: np.round(np.nan_to_num(v), 3).tolist()
                  for k, v in faces.items()},
        "region_table": region_rows,
    }
    return summary, export


# --------------------------------------------------------------------------
# Segment map
# --------------------------------------------------------------------------


def run_segment_map(df, vocab, out, args, videos=None):
    vectors, keep, svd, tfidf = segment_vectors(df, vocab, dims=args.dims)
    seg = df[keep].reset_index(drop=True)
    m = train_som(vectors, args.seg_rows, args.seg_cols, epochs=args.seg_epochs,
                  batch_size=128, n_regions=args.seg_regions)
    rows, cols = m.rows, m.cols
    node = m.node
    regions = m.clusters
    labels = vocab["label"].to_numpy()

    # Prototype term weights: the SVD maps each prototype back onto terms.
    proto = m.weights.reshape(rows * cols, -1) @ svd.components_
    top_terms = [[labels[j] for j in np.argsort(-proto[n])[:8]]
                 for n in range(rows * cols)]

    # Region labels by keyness: mean TF-IDF inside the region minus overall.
    overall = np.asarray(tfidf.mean(axis=0)).ravel()
    region_of_seg = regions.ravel()[node]
    region_rows, centre_labels = [], []
    centres = region_centres(regions)
    for k in np.unique(regions):
        mask = region_of_seg == k
        if mask.sum() == 0:
            continue
        key = np.asarray(tfidf[mask].mean(axis=0)).ravel() - overall
        terms = [labels[j] for j in np.argsort(-key)[:10]]
        region_rows.append({
            "region": int(k), "cells": int((regions == k).sum()),
            "segments": int(mask.sum()),
            "videos": int(seg.loc[mask, "video_id"].nunique()),
            "key_terms": ", ".join(terms),
        })
        centre_labels.append((centres[int(k)], "\n".join(terms[:3])))
    pd.DataFrame(region_rows).to_csv(out / "segment_regions.csv", index=False)
    seg_out = seg[["segment_id", "video_id", "type", "month", "halfyear", "title", "url"]].copy()
    seg_out["row"], seg_out["col"] = m.bmu[:, 0], m.bmu[:, 1]
    seg_out["region"] = region_of_seg
    seg_out.to_csv(out / "segment_cells.csv", index=False)

    hits = m.hits.astype(float)
    hits[hits == 0] = np.nan
    plots.single(out / "10_segment_map_regions.png", m.umatrix,
                 "Segment map: regions with their key terms (shading = U-matrix)",
                 "mean distance to neighbours", cmap=plots.GREY, regions=regions,
                 labels=centre_labels)
    plots.single(out / "11_segment_umatrix.png", m.umatrix,
                 "Segment map, U-matrix: dark = borders between topics",
                 "mean distance to neighbours", cmap=plots.GREY, regions=regions)
    plots.single(out / "12_segment_hits.png", hits,
                 "Segment map, hit map: segments per cell",
                 "segments", regions=regions)
    plots.training_curves(out / "13_segment_training.png", m.qe, m.te,
                          "Segment map training")

    n_nodes = rows * cols

    def per_node(mask):
        counts = np.zeros(n_nodes)
        np.add.at(counts, node[mask], 1)
        return counts

    all_counts = per_node(np.ones(len(seg), bool))
    sparse_cell = all_counts < 3

    # Periods: where on the map each month / half-year talks more than usual.
    month_maps, month_titles, month_export = [], [], {}
    for month, group in seg.groupby(args.period):
        if len(group) < 500:
            continue
        part = per_node(seg[args.period].eq(month).to_numpy())
        face = log_ratio(part, all_counts, part.sum(), all_counts.sum(), prior=3)
        face[sparse_cell] = np.nan
        month_maps.append(face.reshape(rows, cols))
        month_titles.append(f"{month}  ({len(group)} segments)")
        month_export[month] = face
    plots.small_multiples(out / "14_segment_periods.png", month_maps, month_titles,
                          f"Segment map by {args.period}: red = over-represented",
                          regions=regions, cbar_label="log2 share vs. whole period")

    # Format (shorts / video / stream): majority per cell, faded by purity.
    types = sorted(seg["type"].dropna().unique())
    counts_by_type = np.stack([per_node(seg["type"].eq(t).to_numpy())
                               for t in types])
    codes = np.where(all_counts > 0, counts_by_type.argmax(axis=0), -1)
    purity = np.where(all_counts > 0,
                      counts_by_type.max(axis=0) / np.maximum(all_counts, 1), 0)
    plots.category_map(out / "15_segment_format.png", codes.reshape(rows, cols),
                       types, "Segment map, format: majority per cell "
                       "(paler = more mixed)", regions=regions,
                       share=purity.reshape(rows, cols))

    # Caption quality check: if punctuated and unpunctuated captions sorted
    # into different areas, the map would be showing transcript quality
    # rather than content.
    punct = per_node(seg["punctuated"].to_numpy()) / np.maximum(all_counts, 1)
    punct[sparse_cell] = np.nan
    base = seg["punctuated"].mean()
    plots.single(out / "16_segment_punctuation.png", punct.reshape(rows, cols),
                 "Segment map, data-quality check: share of punctuated captions",
                 f"share punctuated (corpus: {base:.0%})", cmap=plots.DIV,
                 regions=regions, norm=plots.TwoSlopeNorm(vcenter=base, vmin=0,
                                                          vmax=1))

    # Video concentration: is a cell a topic, or one long livestream?
    top_video = (seg.assign(node=node).groupby("node")["video_id"]
                 .agg(lambda v: v.value_counts(normalize=True).iloc[0]))
    concentration = np.full(n_nodes, np.nan)
    concentration[top_video.index] = top_video.to_numpy()
    concentration[sparse_cell] = np.nan
    plots.single(out / "19_segment_video_concentration.png",
                 concentration.reshape(rows, cols),
                 "Segment map, share of the cell's segments that come from its "
                 "single largest video", "share from one video", regions=regions)

    # Component planes: one term's weight in every prototype.
    terms = [t for t in args.terms if t in vocab.index]
    planes = [proto[:, vocab.index.get_loc(t)].reshape(rows, cols) for t in terms]
    plots.small_multiples(out / "17_segment_component_planes.png", planes,
                          [vocab.at[t, "label"] for t in terms],
                          "Segment map, component planes: weight of a term in "
                          "each prototype", cmap=plots.SEQ, symmetric=False,
                          regions=regions, cbar_label="term weight", ncols=4)

    views_map = None
    if videos is not None:
        per_video = videos.set_index("id")["viewCount"]
        log_views = np.log10(seg["video_id"].map(per_video).astype(float) + 1)
        sums = np.zeros(n_nodes)
        ok = log_views.notna().to_numpy()
        np.add.at(sums, node[ok], log_views[ok])
        views_map = sums / np.maximum(per_node(ok), 1)
        views_map[sparse_cell] = np.nan
        plots.single(out / "18_segment_views.png", views_map.reshape(rows, cols),
                     "Segment map, metric map: mean log10 views of the videos "
                     "the segments come from", "log10 views", regions=regions)

    # Nearest real segments per cell, for reading what a cell is about.
    dist = np.linalg.norm(vectors - m.weights.reshape(n_nodes, -1)[node], axis=1)
    examples = {}
    for i in np.argsort(dist):
        lst = examples.setdefault(int(node[i]), [])
        if len(lst) < 3:
            lst.append({"title": str(seg.at[i, "title"])[:120],
                        "url": seg.at[i, "url"], "type": seg.at[i, "type"],
                        "month": seg.at[i, args.period],
                        "text": str(seg.at[i, "clean"])[:280]})

    summary = {
        "segments": len(seg), "rows": rows, "cols": cols,
        "quantisation_error": round(m.final_qe, 4),
        "topographic_error_pct": round(m.final_te, 2),
        "empty_cells": int((m.hits == 0).sum()),
    }

    def grid(values):
        return np.round(np.nan_to_num(np.asarray(values, float), nan=-99), 3) \
                 .reshape(rows, cols).tolist()

    export = {
        "rows": rows, "cols": cols, "regions": regions.tolist(),
        "umatrix": np.round(m.umatrix, 4).tolist(), "hits": m.hits.tolist(),
        "top_terms": [top_terms[r * cols:(r + 1) * cols] for r in range(rows)],
        "months": {k: grid(v) for k, v in month_export.items()},
        "types": types, "type_counts": counts_by_type.astype(int).T
                 .reshape(rows, cols, -1).tolist(),
        "punctuated": grid(punct),
        "concentration": grid(concentration),
        "views": grid(views_map) if views_map is not None else None,
        "planes": {vocab.at[t, "label"]: np.round(p, 4).tolist()
                   for t, p in zip(terms, planes)},
        "examples": {str(k): v for k, v in examples.items()},
        "region_table": region_rows,
    }
    return summary, export


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("segments", nargs="+", type=Path, help="*_SEG.csv files")
    parser.add_argument("--videos", nargs="*", type=Path, default=[],
                        help="matching *_DATA.csv files, for view counts")
    parser.add_argument("--name", help="output name (default: first file's run name)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("exploratory/som/output"))
    parser.add_argument("--words", type=int, default=3000)
    parser.add_argument("--dims", type=int, default=200)
    parser.add_argument("--word-rows", type=int, default=22)
    parser.add_argument("--word-cols", type=int, default=32)
    parser.add_argument("--word-regions", type=int, default=24)
    parser.add_argument("--seg-rows", type=int, default=30)
    parser.add_argument("--seg-cols", type=int, default=40)
    parser.add_argument("--seg-regions", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seg-epochs", type=int, default=40)
    parser.add_argument("--terms", nargs="*", default=DEFAULT_TERMS,
                        help="lower-cased terms for the component planes")
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--period", choices=["month", "halfyear"], default=None,
                        help="time slices for the faces (default: month for up "
                             "to 12 months, else halfyear)")
    parser.add_argument("--drop-duplicates", action="store_true",
                        help="leave out segments re-used from another upload")
    parser.add_argument("--drop-roles", nargs="*", default=[],
                        help="segment roles to leave out, e.g. outro weather")
    args = parser.parse_args()

    args.name = args.name or args.segments[0].name.removesuffix("_SEG.csv")
    out = args.output_dir / args.name
    out.mkdir(parents=True, exist_ok=True)

    df = load_segments(args.segments, drop_duplicates=args.drop_duplicates,
                       drop_roles=args.drop_roles)
    if args.period is None:
        args.period = "month" if df["month"].nunique() <= 12 else "halfyear"
    df = lemmatize_segments(df, cache=out / "lemmas.csv", n_process=args.n_process)
    vocab = vocabulary(df)
    videos = None
    if args.videos:
        videos = pd.concat([pd.read_csv(p, usecols=["id", "viewCount"])
                            for p in args.videos], ignore_index=True)
        # A video can appear in two half-year files (scraped twice); keep the
        # row from the file listed last, which is the more recent view count.
        videos = videos.drop_duplicates("id", keep="last")

    word_summary, word_export = run_word_map(df, vocab, out, args)
    seg_summary, seg_export = run_segment_map(df, vocab, out, args, videos)

    summary = {"name": args.name, "segments_in": len(df), "vocabulary": len(vocab),
               "period": args.period, "drop_duplicates": args.drop_duplicates,
               "drop_roles": args.drop_roles,
               "word_map": word_summary, "segment_map": seg_summary}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "explorer_data.json").write_text(json.dumps(
        {"summary": summary, "word": word_export, "segment": seg_export},
        ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
