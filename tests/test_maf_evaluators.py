"""Unit tests for `ds_crew.maf.evaluators`'s two deterministic checks.

Both operate purely on JSON strings shaped like `EvaluationBundle`/
`ExplanationBundle.model_dump_json()` -- no fakes needed beyond that.
"""

from __future__ import annotations

import json

from ds_crew.maf.evaluators import (
    find_leakage_suspicions,
    find_ungrounded_model_mentions,
    format_warning_block,
)


def _bundle(*reports: dict) -> str:
    return json.dumps({"reports": list(reports)})


# ----------------------------------------------------------------------
# find_leakage_suspicions
# ----------------------------------------------------------------------


def test_flags_every_model_with_leakage_suspicion():
    bundle = _bundle(
        {"model_name": "xgboost", "leakage_suspicion": True, "notes": "too good to be true"},
        {"model_name": "knn", "leakage_suspicion": False, "notes": ""},
    )
    findings = find_leakage_suspicions(bundle)
    assert len(findings) == 1
    assert "xgboost" in findings[0]
    assert "too good to be true" in findings[0]


def test_no_findings_when_nothing_is_flagged():
    bundle = _bundle({"model_name": "xgboost", "leakage_suspicion": False, "notes": ""})
    assert find_leakage_suspicions(bundle) == []


def test_degrades_to_empty_on_malformed_json():
    assert find_leakage_suspicions("not json") == []
    assert find_leakage_suspicions(None) == []
    assert find_leakage_suspicions("") == []
    assert find_leakage_suspicions(json.dumps({"no_reports_key": True})) == []


# ----------------------------------------------------------------------
# find_ungrounded_model_mentions
# ----------------------------------------------------------------------


def test_flags_an_evaluated_but_unexplained_model_mentioned_by_name():
    evaluation = _bundle(
        {"model_name": "xgboost", "leakage_suspicion": False},
        {"model_name": "lightgbm", "leakage_suspicion": False},
    )
    explanation = _bundle({"model_name": "xgboost"})
    narration = "I recommend lightgbm, which performed best on the leaderboard."

    findings = find_ungrounded_model_mentions(narration, evaluation, explanation)

    assert len(findings) == 1
    assert "lightgbm" in findings[0]


def test_does_not_flag_a_model_that_was_actually_explained():
    evaluation = _bundle({"model_name": "xgboost", "leakage_suspicion": False})
    explanation = _bundle({"model_name": "xgboost"})
    narration = "xgboost is the recommended model."

    assert find_ungrounded_model_mentions(narration, evaluation, explanation) == []


def test_does_not_flag_an_unexplained_model_the_narration_never_mentions():
    evaluation = _bundle(
        {"model_name": "xgboost", "leakage_suspicion": False},
        {"model_name": "lightgbm", "leakage_suspicion": False},
    )
    explanation = _bundle({"model_name": "xgboost"})
    narration = "xgboost is the recommended model."

    assert find_ungrounded_model_mentions(narration, evaluation, explanation) == []


def test_word_boundary_avoids_a_false_positive_substring_match():
    # "knn" must not match inside an unrelated word like "unknown".
    evaluation = _bundle(
        {"model_name": "xgboost", "leakage_suspicion": False},
        {"model_name": "knn", "leakage_suspicion": False},
    )
    explanation = _bundle({"model_name": "xgboost"})
    narration = "The outcome for the unknown rows was inspected manually."

    assert find_ungrounded_model_mentions(narration, evaluation, explanation) == []


def test_degrades_to_empty_on_malformed_json_or_empty_narration():
    assert find_ungrounded_model_mentions("", "not json", "not json") == []
    assert find_ungrounded_model_mentions("xgboost is great", None, None) == []


# ----------------------------------------------------------------------
# format_warning_block
# ----------------------------------------------------------------------


def test_format_warning_block_lists_every_finding():
    block = format_warning_block(["finding one", "finding two"])
    assert "finding one" in block
    assert "finding two" in block
    assert "AUTOMATED SAFETY CHECK" in block
