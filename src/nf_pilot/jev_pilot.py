"""Score hand-coded pilot segments with Jev and compare with the hand coding.

Usage (from the repo root, with TYPESAFE_API_KEY in .env):

    python -m src.nf_pilot.jev_pilot data/annotations/niusde_2025_H1_pilot_60.xlsx

Only rows with a value in ``relevant`` are scored. Each codebook item is one
Noul question; the segment's neighbours are passed as context but not coded.
Writes to eval/:
    {sheet}_jev.csv         one row per segment and item: human code, Jev p
    {sheet}_jev_report.md   agreement per item and every disagreement
"""

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .codebook import RELEVANCE_QUESTION, load_items

YES, NO = 0.7, 0.3

CONTEXT_NOTE = (
    " Judge only `segment`. `context_before` and `context_after` are the "
    "surrounding transcript and only help to understand it."
)


def build_questions(items):
    """One Noul per item, worded against the `segment` field of the state."""
    from typesafe_sdk import Noul

    questions = {"relevant": RELEVANCE_QUESTION, **items}
    return {
        item: Noul(instructions=text.replace("the segment", "`segment`")
                   + CONTEXT_NOTE)
        for item, text in questions.items()
    }


def state_for(row):
    return {
        "video_title": row["title"],
        "context_before": row["context_before"] or "",
        "segment": row["text"],
        "context_after": row["context_after"] or "",
    }


def score_rows(rows, questions, client, model=None):
    """Return a long DataFrame of Jev probabilities, and the token usage."""
    records, tokens = [], 0
    for row in rows.to_dict("records"):
        response = client.system_one(state_for(row), questions, model=model)
        tokens += response.usage.input_tokens + response.usage.output_tokens
        for item, answer in response.nouls.items():
            records.append({"segment_id": row["segment_id"], "item": item,
                            "jev_p": answer.noul})
    return pd.DataFrame(records), tokens


def compare(sheet_rows, scores, items):
    """Join hand codes to Jev scores and classify each pair."""
    human = sheet_rows.melt(
        id_vars=["segment_id", "notes"], value_vars=["relevant", *items],
        var_name="item", value_name="human",
    )
    merged = human.merge(scores, on=["segment_id", "item"])
    # Items are only coded on relevant segments.
    merged = merged[merged["human"].notna()].copy()
    merged["human"] = merged["human"].astype(int)
    merged["jev"] = (merged["jev_p"] >= 0.5).astype(int)
    merged["status"] = "agree"
    merged.loc[merged["jev_p"].between(NO, YES, inclusive="neither"), "status"] = "uncertain"
    merged.loc[(merged["status"] == "agree") & (merged["human"] != merged["jev"]),
               "status"] = "disagree"
    return merged


def report(merged, sheet_rows, items, tokens):
    order = ["relevant", *items]
    per_item = (
        merged.groupby("item")
        .agg(n=("human", "size"), human_yes=("human", "sum"),
             jev_yes=("jev", "sum"),
             agree=("status", lambda s: (s == "agree").sum()),
             uncertain=("status", lambda s: (s == "uncertain").sum()),
             disagree=("status", lambda s: (s == "disagree").sum()))
        .reindex(order).dropna()
        .astype(int)
    )
    lines = [
        f"# Jev pre-test: {len(sheet_rows)} segments",
        "",
        f"Jev answer counted as yes at p ≥ 0.5, as uncertain for "
        f"{NO} < p < {YES}. Tokens used: {tokens:,}.",
        "",
        f"Overall: {(merged['status'] == 'agree').mean():.0%} agree, "
        f"{(merged['status'] == 'uncertain').mean():.0%} uncertain, "
        f"{(merged['status'] == 'disagree').mean():.0%} confident disagreements "
        f"({len(merged)} codes).",
        "",
        "## Per item",
        "",
        "| item | " + " | ".join(per_item.columns) + " |",
        "|---" * (len(per_item.columns) + 1) + "|",
        *(f"| {item} | " + " | ".join(str(v) for v in row) + " |"
          for item, row in per_item.iterrows()),
        "",
        "## Disagreements and uncertain cases",
        "",
    ]
    texts = sheet_rows.set_index("segment_id")
    for item in order:
        cases = merged[(merged["item"] == item) & (merged["status"] != "agree")]
        if cases.empty:
            continue
        lines.append(f"### {item}")
        for case in cases.itertuples():
            text = texts.loc[case.segment_id, "text"]
            lines.append(
                f"- `{case.segment_id}`: you {case.human}, Jev {case.jev_p:.2f} "
                f"({case.status}). {text[:200]}…"
            )
            if isinstance(case.notes, str) and case.notes.strip():
                lines.append(f"  - Your note: {case.notes.strip()}")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sheet", type=Path, help="coded pilot .xlsx")
    parser.add_argument("--codebook", default="generic_frames.md")
    parser.add_argument("--model", default=None, help="default: jev-latest")
    parser.add_argument("--output-dir", type=Path, default=Path("eval"))
    args = parser.parse_args()

    load_dotenv()
    from typesafe_sdk import TypeSafeClient

    sheet = pd.read_excel(args.sheet, sheet_name="coding")
    coded = sheet[sheet["relevant"].notna()].fillna({"context_before": "",
                                                    "context_after": ""})
    items = load_items(args.codebook)
    print(f"Scoring {len(coded)} coded segments x {len(items) + 1} questions with Jev …")

    with TypeSafeClient() as client:
        scores, tokens = score_rows(coded, build_questions(items), client, args.model)

    merged = compare(coded, scores, items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.sheet.stem
    merged.to_csv(args.output_dir / f"{stem}_jev.csv", index=False)
    report_path = args.output_dir / f"{stem}_jev_report.md"
    report_path.write_text(report(merged, coded, items, tokens), encoding="utf-8")
    print(f"Report -> {report_path}")


if __name__ == "__main__":
    main()
