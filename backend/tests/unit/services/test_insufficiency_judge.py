"""Unit tests for insufficiency judge."""

from backend.app.services.insufficiency_judge import InsufficiencyJudge


def test_parse_decision_accepts_boolean_fields():
    judge = InsufficiencyJudge()
    decision = judge._parse_decision(
        '{"is_sufficient": false, "needs_web_fallback": true, "confidence": 0.8, "reason": "missing info", "missing_information": ["x"], "risk": "low"}'
    )
    assert decision is not None
    assert decision.is_sufficient is False
    assert decision.needs_web_fallback is True
    assert decision.confidence == 0.8


def test_parse_decision_coerces_string_booleans():
    judge = InsufficiencyJudge()
    decision = judge._parse_decision(
        '{"is_sufficient": "false", "needs_web_fallback": "true", "confidence": 0.6, "reason": "missing", "missing_information": [], "risk": "medium"}'
    )
    assert decision is not None
    assert decision.is_sufficient is False
    assert decision.needs_web_fallback is True


def test_parse_decision_rejects_inconsistent_flags():
    judge = InsufficiencyJudge()
    decision = judge._parse_decision(
        '{"is_sufficient": true, "needs_web_fallback": true, "confidence": 0.9, "reason": "bad", "missing_information": [], "risk": "low"}'
    )
    assert decision is None
