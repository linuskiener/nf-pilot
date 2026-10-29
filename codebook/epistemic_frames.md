# Epistemic dimensions (v0, draft)

These categories capture *how* a segment claims to know things, rather than
*what* it is about. They target the concerns of this project: propagandistic
and biased reporting, and opinion mixed into factual reporting. All
categories are a first proposal to be refined on pilot data.

**Unit of analysis:** one segment. Unless stated otherwise, items are binary
(`1` present / `0` absent), and several can apply to the same segment.

## E1 Mode of speech (single choice)
| code | Label | Description |
|---|---|---|
| E1-REP | Reporting | Mainly states events, facts or quotes without evaluation. |
| E1-OPI | Opinion / commentary | Mainly evaluates, argues or judges. |
| E1-MIX | Mixed | Factual claims and evaluation are interwoven in a way that does not mark which is which. **This is the main category of interest.** |
| E1-OTH | Other | Greeting, ads, merchandise, small talk, technical remarks. Also serves as the *irrelevant* class for the relevance filter. |

## E2 Sourcing and evidence
| id | Item |
|---|---|
| E2a | A named, verifiable source is given (person, institution, document, study). |
| E2b | Vague or anonymous sources are used ("many say", "insiders report", "studies show"). |
| E2c | Personal experience or "common sense" is put forward as evidence. |
| E2d | Mainstream media or official sources are cited *in order to be discredited*. |

## E3 Certainty
| id | Item |
|---|---|
| E3a | Claims are hedged or marked as uncertain ("possibly", "it is unclear whether"). |
| E3b | Contested or unverified claims are presented as certain or self-evident. |
| E3c | Rhetorical questions are used to imply a claim without stating it ("Is it a coincidence that …?"). |

## E4 Epistemic authority and distrust
| id | Item |
|---|---|
| E4a | Elites, government, scientists or "the media" are portrayed as deceiving or hiding the truth. |
| E4b | A hidden agenda or coordinated plan behind events is suggested (conspiratorial framing). |
| E4c | The outlet or speaker positions itself as the one telling "what others won't". |

## E5 Propaganda techniques
Based on the technique inventory of Da San Martino et al. (2019/2020; SemEval-2020
Task 11), reduced to the ones expected to be frequent and recognisable in
spoken German. Each technique is binary.

| id | Technique |
|---|---|
| E5a | Loaded language (strongly emotional wording) |
| E5b | Name calling / labelling |
| E5c | Exaggeration or minimisation |
| E5d | Appeal to fear or prejudice |
| E5e | Doubt (questioning credibility without evidence) |
| E5f | Black-and-white fallacy |
| E5g | Causal oversimplification |
| E5h | Whataboutism |
| E5i | Flag-waving (appeal to national or group identity) |
| E5j | Slogans / thought-terminating clichés |

## Open questions
- Whether E1 is coded at segment level or at sentence level and then aggregated:
  sentence level is more precise but multiplies the labelling cost about tenfold.
- E5 may be too fine-grained for reliable hand coding on a small sample.
  A fallback is a single binary "propaganda technique present" item, with the
  technique only as a secondary label.
