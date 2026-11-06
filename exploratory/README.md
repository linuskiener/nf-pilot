# Exploratory analyses

Everything under `exploratory/` is for looking at the corpus, not for
measuring frames. It reads the outputs of the pipeline in `src/nf_pilot/`
(the `*_SEG.csv` segment tables) and never writes back into it. Results here
are not used by the labelling, evaluation or Jev steps.

- `som/`: self-organizing maps of the transcripts (word map and segment map).
