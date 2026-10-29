# Generic news frames (v0)

Source: Semetko, H. A., & Valkenburg, P. M. (2000). Framing European politics:
A content analysis of press and television news. *Journal of Communication*,
50(2), 93–109.

**Unit of analysis:** one segment (about 200 words of transcript, see
`src/nf_pilot/segmentation.py`). The original items are answered per news
story; applying them to segments is an adaptation that the gold standard
should validate.

**Coding:** each item is answered `1` (yes) or `0` (no). A frame counts as
present when at least one of its items is `1`; the item sum gives its
strength. Frames are not mutually exclusive.

Item wording follows the original as closely as possible. Check it against the
paper before the codebook is frozen.

## Attribution of responsibility (`RESP`)
| id | Item |
|---|---|
| RESP1 | Does the segment suggest that some level of government has the ability to alleviate the problem? |
| RESP2 | Does the segment suggest that some level of government is responsible for the issue/problem? |
| RESP3 | Does the segment suggest solution(s) to the problem/issue? |
| RESP4 | Does the segment suggest that an individual (or group of people in society) is responsible for the issue/problem? |
| RESP5 | Does the segment suggest the problem requires urgent action? |

## Human interest (`HUM`)
| id | Item |
|---|---|
| HUM1 | Does the segment provide a human example or "human face" on the issue? |
| HUM2 | Does the segment employ adjectives or personal vignettes that generate feelings of outrage, empathy/caring, sympathy, or compassion? |
| HUM3 | Does the segment emphasize how individuals and groups are affected by the issue/problem? |
| HUM4 | Does the segment go into the private or personal lives of the actors? |
| ~~HUM5~~ | *Visual information item: dropped, transcripts carry no visuals.* |

## Conflict (`CONF`)
| id | Item |
|---|---|
| CONF1 | Does the segment reflect disagreement between parties, individuals, groups or countries? |
| CONF2 | Does one party, individual, group or country reproach another? |
| CONF3 | Does the segment refer to two sides or to more than two sides of the problem or issue? |
| CONF4 | Does the segment refer to winners and losers? |

## Morality (`MOR`)
| id | Item |
|---|---|
| MOR1 | Does the segment contain any moral message? |
| MOR2 | Does the segment make reference to morality, God, and other religious tenets? |
| MOR3 | Does the segment offer specific social prescriptions about how to behave? |

## Economic consequences (`ECON`)
| id | Item |
|---|---|
| ECON1 | Is there a mention of financial losses or gains now or in the future? |
| ECON2 | Is there a mention of the costs or degree of expense involved? |
| ECON3 | Is there a reference to economic consequences of pursuing or not pursuing a course of action? |

## Open questions
- Segment vs. whole video as the unit: streams run about two hours, so whole-video
  coding is not feasible, but frames may build up across neighbouring segments.
- Whether to keep binary items (comparable with the literature) or also ask
  the LLM for a confidence score per item (useful for the cascade in step 4).
