"""Regression tests for citation validity and rationale visibility."""

from runner import assess, rationale_visible


def test_valid_stance_with_only_annotated_citations() -> None:
    score = assess('{"label":"SUPPORT","citations":["2"]}', {"1", "2"}, {"2"}, "SUPPORT")
    assert score["correct"] and score["grounded"] and score["correct_citations"] == 1


def test_invalid_attempts_remain_in_citation_denominator() -> None:
    score = assess('{"label":"SUPPORT","citations":["2","2","3"]}', {"2"}, {"2"}, "SUPPORT")
    assert score["status"] == "invalid" and not score["grounded"]
    assert score["attempted"] == 3 and score["correct_citations"] == 2


def test_citation_requirement_depends_on_context() -> None:
    raw = '{"label":"CONTRADICT","citations":[]}'
    assert assess(raw, set(), {"2"}, "CONTRADICT")["correct"]
    assert assess(raw, {"2"}, {"2"}, "CONTRADICT")["status"] == "invalid"
    raw = '{"label":"INSUFFICIENT","citations":[]}'
    assert assess(raw, {"2"}, {"2"}, "CONTRADICT")["abstained"]


def test_rationale_requires_every_sentence_in_at_least_one_set() -> None:
    sentences = {"2": ["one sentence", "second sentence", "other sentence"]}
    evidence = {"2": [[0, 1], [2]]}
    assert not rationale_visible({"id": "2", "body": "one sentence"}, evidence, sentences)
    assert rationale_visible(
        {"id": "2", "body": "one\nsentence second sentence"}, evidence, sentences
    )
