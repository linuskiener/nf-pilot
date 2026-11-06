"""Clean and segment one or more scraper exports.

Usage (from the repo root):

    python -m src.nf_pilot.prepare_corpus data/raw/niusde_*_DATA.csv

Pass all half-years of a channel at once: re-used clips are searched across
all given files (a Short from early July can come from a June stream).

Writes to data/processed/, per input file:
    {RUN_NAME}_TRANS_trans.csv   one row per video: clean transcript, raw
                                 transcript, caption info, name corrections,
                                 quality columns and derived-clip pointer
    {RUN_NAME}_SEG.csv           one row per segment, with segment_role and
                                 duplicate_of
and once for the whole call:
    cleaning_report.md           what was corrected and flagged
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from .boilerplate import flag_segments
from .cleaning import load_dataframe
from .derivation import find_sources, segment_containment
from .quality import quality_columns
from .segmentation import segment_dataframe


def run_name(path):
    return Path(path).name.removesuffix("_DATA.csv")


def report(videos, segments, path):
    corrections = Counter()
    for s in videos["normalized"].fillna(""):
        for part in filter(None, s.split("; ")):
            k, v = part.rsplit(":", 1)
            corrections[k] += int(v)
    lines = ["# Cleaning report", ""]
    lines.append(f"Files: {', '.join(sorted(videos['run'].unique()))}  ")
    lines.append(f"Videos: {len(videos)}, without transcript: "
                 f"{(videos['transcript'] == '').sum()}, segments: {len(segments)}")
    lines += ["", "## Caption tracks", "",
              videos.groupby(["run", "caption_type"], dropna=False).size()
              .rename("videos").to_frame().to_markdown(), ""]
    month = videos["date"].dt.strftime("%Y-%m")
    lines += ["## Share of punctuated transcripts by month", "",
              videos[videos["transcript"] != ""].groupby([month, "type"])["punctuated"]
              .mean().unstack().round(2).to_markdown(), ""]
    lines += ["## Name corrections", "",
              "| canonical | replacements |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in corrections.most_common()]
    lines += ["", "\"März (ambiguous)\" counts occurrences of \"März\" left as is "
              "that follow neither a person cue nor a temporal cue.", ""]
    lines += ["## Re-used clips (video level)", "",
              pd.crosstab(videos["type"], videos["derived_confidence"]).to_markdown(), ""]
    high = videos[videos["derived_confidence"] == "high"]
    if len(high):
        lines += ["Lag between source and clip for high-confidence clips (days): "
                  f"median {high['derived_lag_days'].median():.1f}, "
                  f"90% within {high['derived_lag_days'].quantile(0.9):.1f}.", ""]
    lines += ["## Segment roles", "",
              pd.crosstab(segments["type"], segments["segment_role"]).to_markdown(), "",
              f"Segments flagged as `duplicate_of` a source: "
              f"{segments['duplicate_of'].notna().sum()}", ""]
    lines += ["## Transcript quality (per video, median)", "",
              videos.groupby("type")[["words_per_minute", "unknown_word_share"]]
              .median().round(3).to_markdown(), ""]
    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="RUN_NAME_DATA.csv exports")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--target-words", type=int, default=200)
    parser.add_argument("--max-words", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in args.inputs:
        df = load_dataframe(path)
        df["run"] = run_name(path)
        frames.append(df)
        print(f"{path.name}: {len(df)} videos")
    videos = pd.concat(frames, ignore_index=True)
    # The same video can appear in two exports; keep the first.
    videos = videos.drop_duplicates("video_id", keep="first").reset_index(drop=True)

    sources = find_sources(videos)
    videos = videos.merge(sources, on="video_id", how="left")
    videos = videos.merge(quality_columns(videos), on="video_id", how="left")

    segments = segment_dataframe(videos, args.target_words, args.max_words)
    segments = flag_segments(segments)
    segments = segment_containment(segments, videos, sources)
    segments["run"] = segments["video_id"].map(dict(zip(videos["video_id"], videos["run"])))

    for run in videos["run"].unique():
        v = videos[videos["run"] == run]
        s = segments[segments["run"] == run].drop(columns="run")
        trans_path = args.output_dir / f"{run}_TRANS_trans.csv"
        v.drop(columns=["cues", "run"]).to_csv(trans_path, index=False)
        seg_path = args.output_dir / f"{run}_SEG.csv"
        s.to_csv(seg_path, index=False)
        print(f"{run}: {len(v)} videos -> {trans_path.name}, {len(s)} segments -> {seg_path.name}")
    report_path = args.output_dir / "cleaning_report.md"
    report(videos, segments, report_path)
    print(f"report -> {report_path}")


if __name__ == "__main__":
    main()
