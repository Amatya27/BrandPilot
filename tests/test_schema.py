"""Unit tests for parsing and validating the model's JSON output."""
import pytest

from brand_assistant_schema import (
    REQUIRED_KEYS,
    extract_json_object,
    is_effectively_empty_output,
    parse_model_json,
    strip_code_fences,
    validate_output,
)


def valid_payload():
    return {
        "brand_summary": "Nike is a global sportswear brand.",
        "likely_target_audience": "Athletes and young adults who follow sport culture.",
        "current_marketing_strategy": ["Athlete partnerships", "Short-form video"],
        "marketing_gaps": ["Few posts aimed at beginner runners"],
        "instagram_post_idea": {
            "hook": "Your first 5K starts here",
            "concept": "A reel following three first-time runners",
            "why_it_closes_the_gap": "Speaks directly to beginners",
        },
        "email_subject": "An idea for reaching new runners",
        "email_body": "Hi team, we noticed a gap in content for beginner runners and would love to "
        "collaborate on a short series that speaks to them directly.",
        "citations": ["[1]"],
    }


def test_strip_code_fences_removes_markdown_and_end_token():
    raw = '```json\n{"a": 1}\n```<end_of_turn>'
    assert strip_code_fences(raw) == '{"a": 1}'


def test_extract_json_object_finds_json_inside_prose():
    raw = 'Here is the answer: {"a": {"b": "has } brace"}} Thanks!'
    assert extract_json_object(raw) == '{"a": {"b": "has } brace"}}'


def test_parse_model_json_raises_when_no_json():
    with pytest.raises(ValueError):
        parse_model_json("the model returned plain text only")


def test_validate_output_accepts_complete_payload():
    assert validate_output(valid_payload()) == []


def test_validate_output_reports_every_missing_key():
    errors = validate_output({})
    for key in REQUIRED_KEYS:
        assert f"Missing required key: {key}" in errors


def test_validate_output_rejects_short_email_body():
    payload = valid_payload()
    payload["email_body"] = "Too short."
    assert any("email_body" in e for e in validate_output(payload))


def test_is_effectively_empty_output():
    assert is_effectively_empty_output({}) is True
    assert is_effectively_empty_output(valid_payload()) is False
