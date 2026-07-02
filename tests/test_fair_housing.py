"""
Unit tests for evaluators/fair_housing.py.

False-positive guard tests are drawn directly from HUD's own published
"acceptable phrase" examples (quiet residential area, parks nearby, gated,
houses of worship nearby, master bedroom, rare find, desirable
neighborhood). A Fair Housing tool that flags its own regulator's
acceptable-phrase examples has a credibility problem, not just a bug --
these tests exist specifically to guard against that.
"""

from evaluators.fair_housing import FairHousingEvaluator


# ── True positives: regex layer should catch these ─────────────────────


def test_no_children_flagged():
    result = FairHousingEvaluator().evaluate("case_1", "This home has no children allowed in the community.")
    assert result.verdict == "fail"
    assert result.details.final_risk == "high"
    assert any(m.category == "familial_status" for m in result.details.tripwire_matches)


def test_adults_only_flagged():
    result = FairHousingEvaluator().evaluate("case_2", "A peaceful adults-only retreat in the heart of the city.")
    assert result.verdict == "fail"
    assert any("adults" in m.matched_phrase.lower() for m in result.details.tripwire_matches)


def test_perfect_for_couples_flagged():
    result = FairHousingEvaluator().evaluate("case_3", "This cozy condo is perfect for couples starting out.")
    assert result.verdict == "fail"
    assert any(m.category == "familial_status" for m in result.details.tripwire_matches)


def test_perfect_for_families_flagged():
    result = FairHousingEvaluator().evaluate("case_4", "A spacious backyard makes this perfect for families.")
    assert result.verdict == "fail"


def test_specific_denomination_church_flagged():
    result = FairHousingEvaluator().evaluate(
        "case_5", "Located just a short walking distance to church for Sunday services."
    )
    assert result.verdict == "fail"
    assert any(m.category == "religion" for m in result.details.tripwire_matches)


def test_catholic_church_nearby_flagged():
    result = FairHousingEvaluator().evaluate("case_6", "Catholic church nearby, along with shopping and dining.")
    assert result.verdict == "fail"
    assert any(m.category == "religion" for m in result.details.tripwire_matches)


def test_english_speakers_only_flagged():
    result = FairHousingEvaluator().evaluate("case_7", "Ideal for English speakers only in a friendly community.")
    assert result.verdict == "fail"
    assert any(m.category == "national_origin" for m in result.details.tripwire_matches)


def test_able_bodied_flagged():
    result = FairHousingEvaluator().evaluate("case_8", "This listing is best suited for able-bodied residents.")
    assert result.verdict == "fail"
    assert any(m.category == "disability" for m in result.details.tripwire_matches)


def test_mature_adults_flagged_for_human_review():
    """Lawful only under HUD's specific senior-housing exemption criteria,
    which text alone can't verify -- must flag, never silently pass."""
    result = FairHousingEvaluator().evaluate("case_9", "A wonderful community for active or mature adults.")
    assert result.verdict == "fail"


# ── The church vs. house-of-worship distinction ─────────────────────────


def test_generic_houses_of_worship_nearby_not_flagged():
    """HUD explicitly lists 'houses of worship nearby' as an ACCEPTABLE
    phrase, distinct from naming a specific denomination's institution."""
    result = FairHousingEvaluator().evaluate(
        "case_10", "Quiet residential area with houses of worship nearby."
    )
    assert result.verdict == "pass"
    assert result.details.tripwire_matches == []


# ── False-positive guards: HUD's own "acceptable" examples ─────────────


def test_master_bedroom_not_flagged():
    result = FairHousingEvaluator().evaluate("case_11", "The master bedroom features a walk-in closet and en-suite bath.")
    assert result.verdict == "pass"


def test_family_room_not_flagged():
    """Bare-word matching on 'family' would falsely flag this. Confirms
    the rule set is phrase-level, not single-token."""
    result = FairHousingEvaluator().evaluate("case_12", "A spacious family room opens onto the covered patio.")
    assert result.verdict == "pass"


def test_quiet_residential_area_not_flagged():
    result = FairHousingEvaluator().evaluate("case_13", "Located in a quiet residential area close to parks.")
    assert result.verdict == "pass"


def test_parks_nearby_not_flagged():
    result = FairHousingEvaluator().evaluate("case_14", "Enjoy parks nearby and easy access to the highway.")
    assert result.verdict == "pass"


def test_gated_not_flagged():
    result = FairHousingEvaluator().evaluate("case_15", "This gated community offers extra peace of mind.")
    assert result.verdict == "pass"


def test_rare_find_and_desirable_neighborhood_not_flagged():
    result = FairHousingEvaluator().evaluate(
        "case_16", "A rare find in one of the area's most desirable neighborhoods."
    )
    assert result.verdict == "pass"


def test_safety_features_not_flagged():
    """Guards against a naive 'safe' tripwire catching legitimate safety
    feature descriptions -- 'safe' is deliberately not in the rule set at
    all for exactly this reason."""
    result = FairHousingEvaluator().evaluate(
        "case_17", "Equipped with modern safety features including a smoke detector system."
    )
    assert result.verdict == "pass"


def test_exclusive_listing_not_flagged():
    """'Exclusive' is deliberately excluded from the regex rule set --
    it's common, benign marketing language most of the time. See module
    docstring for why it's deferred to the judge layer instead."""
    result = FairHousingEvaluator().evaluate("case_18", "An exclusive listing in a sought-after enclave.")
    assert result.verdict == "pass"


# ── No-judge-available behavior ─────────────────────────────────────────


def test_no_judge_available_is_explicit_not_silent():
    """When the regex layer finds nothing and no judge_client is passed,
    the evaluator returns verdict=pass -- but this test exists to make
    that behavior explicit and intentional, not an accidental side effect
    of judge_client defaulting to None. If this test starts failing after
    a judge is wired in, that's expected -- update it to reflect the new
    escalation path rather than deleting it."""
    result = FairHousingEvaluator().evaluate("case_19", "A lovely home with a spacious backyard.")
    assert result.verdict == "pass"
    assert result.details.judge_escalation is None
    assert result.evaluator_type == "deterministic"
    assert result.model_used is None
    assert result.cost_usd is None
