# Issue-specific frames (v0: procedure, no frames yet)

Issue-specific frames cannot be fixed in advance; they have to be derived
from the material. The proposal follows the logic of Matthes & Kohring
(2008): a frame is a recurring pattern of Entman's (1993) four frame
elements, rather than a label chosen intuitively.

## Procedure
1. **Pick issues.** Cluster the relevant segments by topic (embeddings and
   clustering, or an LLM topic label). Choose 3–5 issues that are frequent
   across outlets, e.g. migration and asylum, energy and climate policy,
   the war in Ukraine, the government coalition, and the public broadcasters.
2. **Code frame elements.** For a sample of about 100 segments per issue,
   extract the four elements (LLM-assisted, human-checked):
   | Element | Question |
   |---|---|
   | Problem definition | What is presented as the problem, and who is affected? |
   | Causal attribution | Who or what is blamed for it? |
   | Moral evaluation | How are the actors and the situation judged? |
   | Treatment recommendation | What should be done, and by whom? |
3. **Find patterns.** Cluster the element combinations, or have an LLM
   propose frame candidates from them, then consolidate by hand into
   3–6 frames per issue with a definition and anchor examples.
4. **Freeze and validate.** Write each frame up in the format below, and
   check it on the gold standard together with the generic frames.

## Frame template
```
### <ISSUE>-<CODE> <Short name>
Definition:        one or two sentences
Problem:           ...
Cause:             ...
Evaluation:        ...
Treatment:         ...
Anchor example:    segment_id + quote
Not this frame if: delimitation from the closest other frame
```

## Research question link
Step 3 compares how well (a) the generic frames and (b) these issue-specific
frames can be coded by an LLM against the gold standard (F1, agreement), and
how much each explains about differences between outlets.
