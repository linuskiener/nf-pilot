# Which Jev primitive for which task

Jev (TypeSafe System One) offers three question types. Each call takes a
`state` (here: the segment, with its title and context) and a set of
questions that are answered in parallel.

| Primitive | Answers | Output | Use when |
|---|---|---|---|
| **Noul** | one yes/no question | `noul`: probability of *yes* (0–1) | the item is binary |
| **Choice** | pick one of up to 255 options | `choice`, `probabilities` per option, `confidence` | exactly one option applies |
| **Score** | place on 2–10 ordered, described levels | `score` (weighted mean level), `probabilities`, `confidence` | there is a meaningful in-between |

Notes from the docs that matter here:
- Choice handles "none of the above" poorly. Check that with a separate Noul first.
- Score levels must be described situations, not numbers, and one Score covers one dimension only.
- Noul returns no confidence. Uncertainty is the distance from 0.5, and the docs suggest 0.7 / 0.3 as default cut-offs.
- Confidence for Choice and Score is 1 when all probability sits on one option. It falls towards 0 as the probability spreads out.

## Mapping for this project

| Pipeline task | Primitive | Why | Routing to the reasoning model |
|---|---|---|---|
| **Relevance** (news/politics vs. ads, greetings, small talk) | Noul | binary, and it runs first on every segment, so the cheapest question matters most | 0.3 < p < 0.7 |
| **Generic frames**: 19 S&V items | Noul, one per item | the items are defined as yes/no, so results stay comparable with the literature | per item: 0.3 < p < 0.7 (to be tuned on the pilot) |
| Generic frame *strength* (derived) | none, computed in code | mean of item probabilities per frame | — |
| **E1 mode of speech**: reporting / opinion / mixed | Choice (only on relevant segments) | the categories are mutually exclusive, and "other" is already handled by relevance | confidence < 0.5 |
| E1 alternative: *degree of evaluation* | Score, 5 levels from "pure reporting" to "pure commentary" | "mixed" is really a middle of a scale, so this gives a continuous measure | confidence < 0.5 |
| **E2–E4** sourcing, certainty, distrust | Noul, one per item | binary and can co-occur | 0.3 < p < 0.7 |
| **E5 propaganda techniques** | Noul per technique, plus Score for loaded language intensity | techniques co-occur; intensity has degrees | 0.3 < p < 0.7 |
| **Issue / topic** of a segment | Noul "about one of these issues?" then Choice over the issue list | follows the docs' advice for "none of the above" | confidence < 0.5 → second Choice or reasoning model |
| **Issue-specific frames** (once defined) | Noul per frame (presence); Choice for the *dominant* frame | frames can co-occur, but a dominant frame is often what is compared | as above |
| **Deriving issue frames** (Entman elements, frame proposals) | none: reasoning LLM | free-text extraction and induction; Jev only answers closed questions | — |

## Open points to test in the pilot
1. **Noul vs. Score for the generic frames.** Noul per item is closest to S&V, but a Score per frame ("not present / mentioned / central") could be cheaper (5 questions instead of 19) and closer to how coders actually perceive frames. Running both on the pilot shows which agrees better with hand coding.
2. **Language of instructions.** The examples are in English. We should test English criteria vs. German criteria on German segments.
3. **State format.** `{"segment": ..., "title": ..., "context_before": ..., "context_after": ...}` as structured state, with the instructions referring to `segment` so that context is read but not coded.
