# nf-pilot

Pilot for (issue-specific) news frame and epistemic frame detection in transcripts of German and Swiss news outlets and podcasts on YouTube.

## Plan
1. Corpus, segmentation and codebook v0
2. Gold standard (hand-coded sample, inter-coder reliability)
3. LLM labelling benchmark (generic vs. issue-specific frames, F1, cost)
4. Efficient cascade (Jev for relevance/topics/simple frames, reasoning model for hard cases, German BERT baseline)

## Layout
- `data/raw/`: original transcripts (not committed)
- `data/processed/`: segmented corpus (not committed)
- `data/annotations/`: hand-coded gold labels
- `codebook/`: frame definitions
- `prompts/`: LLM prompt templates
- `src/nf_pilot/`: segmentation, labelling and evaluation code
- `notebooks/`: exploration
- `eval/`: results and metrics

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys, never commit .env
```

## Preparing a corpus
Put the scraper export (e.g. `niusde_2025_H1_DATA.csv`) in `data/raw/`, then:
```bash
python -m src.nf_pilot.prepare_corpus data/raw/niusde_2025_H1_DATA.csv
```
This writes a cleaned per-video table (`*_TRANS_trans.csv`) and a per-segment table (`*_SEG.csv`) to `data/processed/`. Segments are about 200 words, end at sentence boundaries where the captions have punctuation, and carry a YouTube link to the exact start time.
