"""Suggest speech-recognition spelling variants of frequent words.

    python -m exploratory.som.suggest_variants exploratory/som/output/niusde_2025_H1

Uses the lemma cache of a SOM run. A rarer term is suggested as a variant of
a frequent one when the two are spelled alike AND are used in the same
contexts (the same co-occurrence vectors the word map is built from). That
second condition is what tells "Habbeck" (used like Habeck) from "Haben"
(spelled like it, used differently). The output is a list to review by hand;
confirmed pairs go into ``src/nf_pilot/resources/name_variants.csv``.
"""

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .represent import ppmi, term_matrix, vocabulary


def suggest(df, min_ratio=0.75, min_cosine=0.35, min_freq_ratio=3.0,
            n_contexts=5000, dims=150):
    vocab = vocabulary(df, min_segments=3, min_videos=2, max_share=0.25)
    x = term_matrix(df, vocab.index, binary=True)
    co = (x.T @ x[:, :n_contexts]).toarray().astype(np.float32)
    np.fill_diagonal(co[:n_contexts], 0)
    vec = normalize(TruncatedSVD(dims, random_state=0).fit_transform(ppmi(co)))
    terms = vocab.index.to_list()
    freq = vocab["tokens"].to_numpy()
    rows = []
    for block in range(0, len(terms), 1024):
        sims = vec[block:block + 1024] @ vec.T
        for k, row in enumerate(sims):
            i = block + k
            # Only look at the 40 most similar and clearly more frequent terms.
            row = np.where(freq >= min_freq_ratio * freq[i], row, -1.0)
            for j in np.argpartition(-row, 40)[:40]:
                cos = float(row[j])
                if cos < min_cosine:
                    continue
                t, u = terms[i], terms[j]
                if abs(len(u) - len(t)) > 3:
                    continue
                ratio = SequenceMatcher(None, t, u).ratio()
                if ratio >= min_ratio:
                    rows.append({
                        "variant": vocab.at[t, "label"],
                        "canonical": vocab.at[u, "label"],
                        "variant_tokens": int(freq[i]),
                        "canonical_tokens": int(freq[j]),
                        "spelling": round(ratio, 2), "context": round(cos, 2),
                    })
    out = pd.DataFrame(rows)
    if len(out):
        out["score"] = (out["spelling"] * out["context"]).round(3)
        out = out.sort_values("score", ascending=False)
        out = out.drop_duplicates("variant")  # best canonical per variant
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="SOM output dir with lemmas.csv")
    args = parser.parse_args()
    lemmas = pd.read_csv(args.run_dir / "lemmas.csv", keep_default_na=False)
    segments = pd.read_csv(args.run_dir / "segment_cells.csv",
                           usecols=["segment_id", "video_id"])
    df = lemmas.merge(segments, how="left", on="segment_id")
    df["video_id"] = df["video_id"].fillna(df["segment_id"].str[:11])
    out = suggest(df)
    path = args.run_dir / "variant_suggestions.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} suggestions -> {path}")
    print(out.head(60).to_string(index=False))


if __name__ == "__main__":
    main()
