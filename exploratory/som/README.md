# Self-organizing maps of the transcripts

Two self-organizing maps (SOMs) for exploring what the channels talk about,
built after the literature in the project thread: the word "galaxy" in
*Play Among Books* (Roman & Alice_ch3n81), the document SOM with hit map and
U-matrix of Bungum & Gambäck (2015), and the visualisations of `torchsom`
(Berthier et al.), which is also the SOM implementation used here.

## Run

```bash
pip install -r exploratory/som/requirements.txt        # once; torch can be the CPU build
python -m src.nf_pilot.prepare_corpus data/raw/niusde_*_DATA.csv   # all half-years at once
python -m exploratory.som.run data/processed/niusde_*_SEG.csv --videos data/raw/niusde_*_DATA.csv --name niusde_all --drop-duplicates --drop-roles outro weather --n-process 4
python -m exploratory.som.explorer exploratory/som/output/niusde_all
```

Several `*_SEG.csv` files can be given at once (e.g. two half-years, or two
channels); they are then placed on one shared map, and `--name` names the run.
On a laptop CPU the whole run takes a few minutes, most of it lemmatisation,
which is cached in `output/<name>/lemmas.csv`. Useful options: `--words`
(size of the word map vocabulary), `--word-rows/--word-cols`,
`--seg-rows/--seg-cols`, `--word-regions/--seg-regions` (k-means regions),
`--terms` (component planes to draw), `--period month|halfyear` (time
slices for the faces; default month up to 12 months, otherwise half-year),
`--drop-duplicates` (leave out segments that `prepare_corpus` found re-used
from a longer upload, so a stream and its Shorts are not counted twice) and
`--drop-roles` (leave out show furniture, e.g. `outro weather`).

`python -m exploratory.som.suggest_variants exploratory/som/output/<name>`
lists spelling variants that the SOM vectors place in the same contexts as a
more frequent word (`variant_suggestions.csv`), as input for the reviewed
list in `src/nf_pilot/resources/name_variants.csv`.

## Design decisions

**Two units of observation.** The book maps *words*: each word is described
by how it is spread over the books of a library, and the SOM puts words with
similar spreads next to each other. That gives the grid of squares filled
with words. The Reuters paper maps *documents*, which is what metadata and,
later, frame labels can be laid over. Both are useful here, so both are
built from the same cleaned vocabulary:

| | Word map | Segment map |
|---|---|---|
| observation | lemma (top 3,000 by spread) | ~200-word segment |
| described by | which words it shares segments with (PPMI) | its own words (TF-IDF) |
| compressed to | 200 SVD dimensions | 200 SVD dimensions |
| default grid | 22 x 32 | 30 x 40 |

**Segments, not videos, as context.** A video is a bad "book" in this
corpus: a Short has 150 words, a livestream 30,000. The existing segmentation
step already cuts transcripts into ~200-word segments, so every context has
about the same weight, and co-occurrence inside a segment means "talked about
in the same stretch".

**Cleaning.** Caption markers (`[Musik]`, `[Applaus]`, `>>`) and broken SRT
time stamps (`00:00:27,830 --> 00:aN:NaN,NaN`, still present in about 1 in 9
segments of the SEG file) are removed. spaCy `de_core_news_sm` lemmatises and
keeps nouns, proper nouns, adjectives and verbs; spaCy's stop words plus a
short list of spoken fillers and light verbs (`corpus.EXTRA_STOPWORDS`) are
dropped. A term has to appear in at least 20 segments of at least 8 videos,
so one stream's transcription errors cannot claim a square.

**Punctuation does not matter here.** Both maps use bags of lemmas, so
sentence boundaries are never used. Name variants from speech recognition
(Habbeck, "Friedrich März") are corrected upstream by
`src/nf_pilot/normalize.py`; the small spaCy model's truncated lemmas
("Migrationswend") are repaired in `corpus.repair_map`.

**SOM settings.** `torchsom.SOM`, rectangular grid, PCA initialisation,
Gaussian neighbourhood with sigma from a quarter of the longer side down to 1,
asymptotic learning-rate decay, Euclidean distance on length-normalised
vectors (equivalent to cosine). Regions are k-means clusters of the trained
prototypes (`SOM.cluster`).

## Outputs (`output/<name>/`)

| file | what it shows |
|---|---|
| `01_word_grid.png/.pdf` | the word map: every square lists its words (most frequent first), `nXeY` = node X with Y words as in the book; grey shading = U-matrix; thick lines = regions. The PDF is vector, zoom in. |
| `02_word_umatrix.png` | distance of each square to its neighbours; dark ridges separate word fields |
| `03_word_hits.png` | words per square |
| `04_word_training.png`, `13_segment_training.png` | quantisation and topographic error per epoch |
| `05_word_faces_period.png`, `05_word_faces_type.png` | the book's "spectral faces": how much more (red) or less (blue) a month or format uses each square's words than the whole corpus, log2, shrunk towards 0 where counts are small |
| `10_segment_map_regions.png` | segment map regions with their key terms |
| `11_segment_umatrix.png`, `12_segment_hits.png` | U-matrix and hit map of the segment map |
| `14_segment_periods.png` | where each month (or half-year) talks more or less than usual |
| `15_segment_format.png` | majority format per square (shorts / stream / video), paler = mixed |
| `16_segment_punctuation.png` | data check: share of punctuated captions per square |
| `19_segment_video_concentration.png` | data check: share of a square's segments from its single largest video |
| `17_segment_component_planes.png` | weight of chosen terms in every prototype |
| `18_segment_views.png` | mean log10 views of the source videos (needs `--videos`) |
| `word_cells.csv`, `word_regions.csv` | every word with its square and region; region summaries |
| `segment_cells.csv`, `segment_regions.csv` | every segment with its square and region (with YouTube link); region summaries |
| `explorer.html` | the interactive atlas: both maps, all layers, word search, closest segments per square |

## Reading the figures (NIUS 2025 H1)

- The word map has topical fields at the edges: public finances and taxes
  top left, energy and climate along the top, parties and coalition top
  right, foreign policy (Trump, Ukraine, Israel) on the right, migration and
  asylum on the left, knife attacks and policing bottom left, media and public
  broadcasting along the bottom. The centre collects general verbs that occur
  everywhere.
- The month layers of the segment map read like a timeline: January and
  February over-represent Habeck, Scholz, the parties and the election;
  March the debt package ("Milliarde, Euro") and the NGO funding question;
  April taxes and pensions; May pensions, the AfD report of the Verfassungsschutz
  and knife crime; June Israel/Iran.
- The segment map is not driven by one stream per square (the concentration
  map stays low almost everywhere), so its regions are topics, not videos.
- Punctuation is a proxy for time, not for quality: all captions in the
  export are auto-generated, and YouTube's recogniser started punctuating in
  March 2025 (0-5 % punctuated in Jan/Feb, 84-100 % from April). The Habeck
  and Scholz regions are 80-90 % unpunctuated because 75-80 % of their
  segments are from January and February. Restricting analyses to punctuated
  transcripts would therefore cut out the election campaign.
