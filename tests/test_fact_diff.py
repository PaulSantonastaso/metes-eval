"""
Unit tests for evaluators/fact_diff.py.

This is the one piece of matching logic in the harness that's easy to get
subtly wrong and hard to notice when it's wrong -- a false "present" hides
a real hallucination, a false "missing" makes good copy look broken. Tests
are organized around the edge cases called out during design: comma
formatting, digit/word number forms, unit synonyms, and the known
half-bath limitation.
"""

import json
from pathlib import Path

import pytest

from evaluators.fact_diff import FactDiffEvaluator, normalize_and_match

GOLDEN_DATASET_PATH = Path(__file__).parent.parent / "data" / "golden_dataset.json"


def load_case(case_id: str) -> dict:
    cases = json.loads(GOLDEN_DATASET_PATH.read_text())
    for case in cases:
        if case["case_id"] == case_id:
            return case
    raise KeyError(f"No case with id {case_id} in golden dataset")


# ── normalize_and_match: number formatting ──────────────────────────────


def test_comma_formatted_number_matches():
    assert normalize_and_match("1850", "This home offers 1,850 square feet of living space.")


def test_bare_number_no_false_positive_on_partial_match():
    # "1850" should not match inside "18505" -- word boundary check.
    assert not normalize_and_match("1850", "Listed under MLS# 18505.")


def test_digit_form_matches_word_form_in_text():
    assert normalize_and_match("3 bed", "This gorgeous three-bedroom home won't last.")


def test_word_form_fact_matches_digit_form_in_text():
    assert normalize_and_match("three bed", "Featuring 3 beds and 2 baths.")


# ── normalize_and_match: unit synonyms ──────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        "3 bed, 2 bath home",
        "3-bedroom, 2-bathroom home",
        "This home has 3 bedrooms and 2 bathrooms",
        "3BR/2BA in a quiet community",
    ],
)
def test_bed_bath_unit_synonyms(text):
    assert normalize_and_match("3 bed", text)
    assert normalize_and_match("2 bath", text)


def test_sqft_unit_synonyms():
    assert normalize_and_match("1850 sqft", "Offering 1,850 square feet of thoughtfully designed space.")
    assert normalize_and_match("1850 sqft", "1850 sq ft of living area.")


def test_garage_unit_synonym():
    assert normalize_and_match("2 garage", "Includes a spacious 2-car garage.")


# ── normalize_and_match: plain phrase facts ─────────────────────────────


def test_plain_phrase_fact_case_insensitive():
    assert normalize_and_match("granite countertops", "The kitchen features Granite Countertops throughout.")


def test_plain_phrase_fact_not_present():
    assert not normalize_and_match("granite countertops", "The kitchen features quartz countertops throughout.")


# ── normalize_and_match: known limitation ───────────────────────────────


def test_half_bath_known_limitation():
    """
    Documents a real gap rather than hiding it: "2.5 bath" as a fact
    string does not match copy phrased as "2 full baths and 1 half bath".
    Fixing this needs a bathroom-specific parser (half_bathrooms is a
    distinct field on PropertyDetails); deferred rather than special-cased
    into the general-purpose matcher. See module docstring.
    """
    assert not normalize_and_match("2.5 bath", "This home offers 2 full baths and 1 half bath.")


# ── FactDiffEvaluator: end-to-end against golden dataset ────────────────


def test_evaluator_pass_when_all_facts_present_and_none_fabricated():
    case = load_case("std_001")
    generated_text = (
        "Welcome to this beautifully maintained 3 bed, 2 bath home offering "
        "1,850 square feet of living space, complete with granite countertops "
        "and a screened lanai perfect for Florida evenings."
    )
    result = FactDiffEvaluator().evaluate(case, generated_text)

    assert result.verdict == "pass"
    assert result.evaluator_name == "fact_diff"
    assert result.evaluator_type == "deterministic"
    assert result.score is None
    assert result.cost_usd is None
    assert result.model_used is None
    assert set(result.details.facts_missing) == set()
    assert set(result.details.facts_fabricated) == set()
    assert len(result.details.facts_present) == 5


def test_evaluator_flags_missing_fact_without_fabrication():
    case = load_case("std_001")
    # Omits "granite countertops" and "screened lanai".
    generated_text = "Welcome to this beautifully maintained 3 bed, 2 bath home offering 1,850 square feet."
    result = FactDiffEvaluator().evaluate(case, generated_text)

    assert result.verdict == "flag"
    assert "granite countertops" in result.details.facts_missing
    assert "screened lanai" in result.details.facts_missing
    assert result.details.facts_fabricated == []


def test_evaluator_fails_on_fabrication_regardless_of_missing_facts():
    case = load_case("adv_missing_001")
    # Fabricates a square footage the input never provided.
    generated_text = "This charming 3 bed townhome offers 2,000 square feet in a fenced yard setting."
    result = FactDiffEvaluator().evaluate(case, generated_text)

    assert result.verdict == "fail"
    assert "2000 square feet" in result.details.facts_fabricated


def test_evaluator_fails_on_fh_tripwire_phrase_leaking_into_copy():
    case = load_case("adv_fh_001")
    generated_text = (
        "This 4 bed, 2 bath home at 2,100 square feet is perfect for families "
        "and offers a covered patio."
    )
    result = FactDiffEvaluator().evaluate(case, generated_text)

    assert result.verdict == "fail"
    assert "perfect for families" in result.details.facts_fabricated


def test_evaluator_verdict_precedence_fail_beats_flag():
    """A case that is both missing a fact AND has a fabrication must fail,
    not flag -- fabrication severity always wins."""
    case = load_case("adv_missing_001")
    # Missing "fenced yard", fabricates a square footage.
    generated_text = "This 3 bed townhome offers 2,000 square feet."
    result = FactDiffEvaluator().evaluate(case, generated_text)

    assert result.verdict == "fail"
    assert "fenced yard" in result.details.facts_missing
    assert "2000 square feet" in result.details.facts_fabricated
