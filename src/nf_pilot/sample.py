"""Draw a stratified sample of segments and write a coding sheet.

Usage (from the repo root):

    python -m src.nf_pilot.sample data/processed/niusde_2025_H1_SEG.csv --n 60

Writes data/annotations/{RUN_NAME}_pilot_{n}.xlsx with one row per segment
and one column per codebook item. Fill each item with 1 (yes) or 0 (no).
"""

import argparse
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from .codebook import RELEVANCE_QUESTION, load_items

CONTEXT_WORDS = 40


def add_context(segments):
    """Add the end of the previous and the start of the next segment."""
    segments = segments.sort_values(["video_id", "position"])
    words = segments["text"].str.split()
    same_video = segments["video_id"].eq(segments["video_id"].shift())
    segments["context_before"] = (
        words.shift().where(same_video).map(
            lambda w: "… " + " ".join(w[-CONTEXT_WORDS:]) if isinstance(w, list) else ""
        )
    )
    same_video = segments["video_id"].eq(segments["video_id"].shift(-1))
    segments["context_after"] = (
        words.shift(-1).where(same_video).map(
            lambda w: " ".join(w[:CONTEXT_WORDS]) + " …" if isinstance(w, list) else ""
        )
    )
    return segments


def stratified_sample(segments, n, min_words=80, seed=42):
    """Sample about n segments, equally across type x punctuated strata.

    At most one segment per video, so no single stream dominates, and spread
    over months by sampling within each stratum at random.
    """
    pool = segments[segments["n_words"] >= min_words]
    pool = pool.sample(frac=1, random_state=seed).drop_duplicates("video_id")
    strata = pool.groupby(["type", "punctuated"])
    per_stratum = -(-n // strata.ngroups)
    sample = strata.head(per_stratum)
    return sample.sample(n=min(n, len(sample)), random_state=seed)


def write_sheet(sample, items, path):
    meta = ["segment_id", "url", "type", "date", "title",
            "context_before", "text", "context_after"]
    sheet = sample[meta].copy()
    sheet.insert(len(meta), "relevant", "")
    for item in items:
        sheet[item] = ""
    sheet["notes"] = ""
    codebook = pd.DataFrame(
        {"item": ["relevant", *items],
         "question": [RELEVANCE_QUESTION
                      + " If 0, leave the other items empty.", *items.values()]}
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, index=False, sheet_name="coding")
        codebook.to_excel(writer, index=False, sheet_name="codebook")

    wb = load_workbook(path)
    ws = wb["coding"]
    widths = {"A": 18, "B": 12, "C": 8, "D": 11, "E": 30,
              "F": 30, "G": 90, "H": 30}
    for col, width in widths.items():
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2, max_col=8):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        row[1].hyperlink = row[1].value
        row[1].value = "open video"
        row[1].font = Font(color="0563C1", underline="single")
    first_item = len(meta) + 1
    last_item = first_item + len(items)
    yes_no = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
    ws.add_data_validation(yes_no)
    for col in range(first_item, last_item + 1):
        letter = ws.cell(row=1, column=col).column_letter
        ws.column_dimensions[letter].width = 8
        yes_no.add(f"{letter}2:{letter}{ws.max_row}")
        ws.cell(row=1, column=col).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "B2"
    wb["codebook"].column_dimensions["B"].width = 110
    wb.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("segments", type=Path, help="RUN_NAME_SEG.csv")
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--codebook", default="generic_frames.md")
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"))
    args = parser.parse_args()

    segments = add_context(pd.read_csv(args.segments))
    sample = stratified_sample(segments, args.n, seed=args.seed)
    items = load_items(args.codebook)

    run_name = args.segments.name.removesuffix("_SEG.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{run_name}_pilot_{args.n}.xlsx"
    write_sheet(sample, items, path)
    print(f"{len(sample)} segments, {len(items)} items -> {path}")
    print(sample.groupby(["type", "punctuated"]).size())


if __name__ == "__main__":
    main()
