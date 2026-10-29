"""Clean and segment one scraper export.

Usage (from the repo root):

    python -m src.nf_pilot.prepare_corpus data/raw/niusde_2025_H1_DATA.csv

Writes to data/processed/:
    {RUN_NAME}_TRANS_trans.csv   one row per video, with a clean transcript
    {RUN_NAME}_SEG.csv           one row per segment
"""

import argparse
from pathlib import Path

from .cleaning import load_dataframe
from .segmentation import segment_dataframe


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="RUN_NAME_DATA.csv export")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--target-words", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args()

    run_name = args.input.name.removesuffix("_DATA.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = load_dataframe(args.input)
    segments = segment_dataframe(videos, args.target_words, args.max_words)

    trans_path = args.output_dir / f"{run_name}_TRANS_trans.csv"
    videos.drop(columns="cues").to_csv(trans_path, index=False)
    seg_path = args.output_dir / f"{run_name}_SEG.csv"
    segments.to_csv(seg_path, index=False)

    without = (videos["transcript"] == "").sum()
    print(f"{len(videos)} videos ({without} without transcript) -> {trans_path}")
    print(f"{len(segments)} segments -> {seg_path}")
    print(segments.groupby("type")["n_words"].describe().round(0))


if __name__ == "__main__":
    main()
