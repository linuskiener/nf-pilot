from types import SimpleNamespace

import pandas as pd

from src.nf_pilot.codebook import load_items
from src.nf_pilot.jev_pilot import build_questions, compare, report, score_rows


class FakeClient:
    """Answers every question with a fixed probability per item."""

    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.states = []

    def system_one(self, state, questions, model=None):
        self.states.append(state)
        nouls = {q: SimpleNamespace(noul=self.probabilities.get(q, 0.1))
                 for q in questions}
        usage = SimpleNamespace(input_tokens=100, output_tokens=10)
        return SimpleNamespace(nouls=nouls, usage=usage)


def test_pre_test_pipeline():
    items = load_items()
    assert len(items) == 19 and "HUM5" not in items
    questions = build_questions(items)
    assert "`segment`" in questions["HUM4"].instructions

    rows = pd.DataFrame([
        {"segment_id": "a_0001", "title": "T", "context_before": "",
         "text": "Text A", "context_after": "", "notes": "unsure about CONF1",
         "relevant": 1, **{i: 0 for i in items}, "CONF1": 1, "HUM4": 1},
        {"segment_id": "b_0002", "title": "T", "context_before": "",
         "text": "Text B", "context_after": "", "notes": None,
         "relevant": 0, **{i: None for i in items}},
    ])
    client = FakeClient({"relevant": 0.9, "CONF1": 0.95, "HUM4": 0.5})
    scores, tokens = score_rows(rows, questions, client)
    assert client.states[0]["segment"] == "Text A"
    assert tokens == 220

    merged = compare(rows, scores, items)
    status = merged.set_index(["segment_id", "item"])["status"]
    assert status[("a_0001", "CONF1")] == "agree"
    assert status[("a_0001", "HUM4")] == "uncertain"
    assert status[("b_0002", "relevant")] == "disagree"
    # Items of an irrelevant segment are not compared.
    assert ("b_0002", "CONF1") not in status.index

    text = report(merged, rows, items, tokens)
    assert "| HUM4 |" in text and "unsure about CONF1" in text
