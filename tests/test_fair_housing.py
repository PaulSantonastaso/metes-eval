"""
Unit tests for evaluators/fair_housing.py.

False-positive guard tests are drawn directly from HUD's own published
"acceptable phrase" examples (quiet residential area, parks nearby, gated,
houses of worship nearby, master bedroom, rare find, desirable
neighborhood). A Fair Housing tool that flags its own regulator's
acceptable-phrase examples has a credibility problem, not just a bug --
these tests exist specifically to guard against that.

Async throughout: evaluate() is a coroutine because the judge escalation
path does real I/O. pytest-asyncio runs in "auto" mode (see pyproject.toml)
so `async def test_...` works without decorators.

`assert_fh_details(result)` narrows EvalResult.details (typed as a union
of FactDiffDetails | JudgeDetails | FairHousingDetails, per the schema
design) down to FairHousingDetails specifically. This isn't just to
satisfy the type checker -- it's a real runtime guarantee that
FairHousingEvaluator always returns FairHousingDetails, which is worth
asserting explicitly rather than assuming.
"""

from evaluators.base import EvalResult, FairHousingDetails, JudgeDetails
from evaluators.fair_housing import FairHousingEvaluator
from judges.base import JudgeCallResult, JudgeClient


def assert_fh_details(result: EvalResult) -> FairHousingDetails:
    assert isinstance(result.details, FairHousingDetails)
    return result.details


class MockJudgeClient(JudgeClient):
    """A fake judge that returns a pre-set risk score, so the escalation
    path's threshold/verdict-mapping logic can be tested without a real
    API call. model_name matches the JudgeClient contract."""

    model_name = "mock-judge-v1"

    def __init__(self, risk_level: float, rationale: str = "mock rationale"):
        self._risk_level = risk_level
        self._rationale = rationale

    async def score_quality(self, mls_summary, headline, property_context) -> JudgeCallResult:
        raise NotImplementedError("not used by fair_housing tests")

    async def score_fair_housing_risk(self, text: str) -> JudgeCallResult:
        details = JudgeDetails(
            rationale=self._rationale,
            criteria_scores={"fair_housing_risk_level": self._risk_level},
            raw_judge_response=f'{{"fair_housing_risk_level": {self._risk_level}}}',
            prompt_version="mock00ab",
        )
        return JudgeCallResult(
            details=details,
            latency_ms=42,
            cost_usd=0.0001,
            model_used=self.model_name,
        )


# ── True positives: regex layer should catch these ─────────────────────


async def test_no_children_flagged():
    result = await FairHousingEvaluator().evaluate("case_1", "This home has no children allowed in the community.")
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert details.final_risk == "high"
    assert any(m.category == "familial_status" for m in details.tripwire_matches)


async def test_adults_only_flagged():
    result = await FairHousingEvaluator().evaluate("case_2", "A peaceful adults-only retreat in the heart of the city.")
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert any("adults" in m.matched_phrase.lower() for m in details.tripwire_matches)


async def test_perfect_for_couples_flagged():
    result = await FairHousingEvaluator().evaluate("case_3", "This cozy condo is perfect for couples starting out.")
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert any(m.category == "familial_status" for m in details.tripwire_matches)


async def test_perfect_for_families_flagged():
    result = await FairHousingEvaluator().evaluate("case_4", "A spacious backyard makes this perfect for families.")
    assert result.verdict == "fail"


async def test_specific_denomination_church_flagged():
    result = await FairHousingEvaluator().evaluate(
        "case_5", "Located just a short walking distance to church for Sunday services."
    )
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert any(m.category == "religion" for m in details.tripwire_matches)


async def test_catholic_church_nearby_flagged():
    result = await FairHousingEvaluator().evaluate("case_6", "Catholic church nearby, along with shopping and dining.")
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert any(m.category == "religion" for m in details.tripwire_matches)


async def test_english_speakers_only_flagged():
    result = await FairHousingEvaluator().evaluate("case_7", "Ideal for English speakers only in a friendly community.")
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert any(m.category == "national_origin" for m in details.tripwire_matches)


async def test_able_bodied_flagged():
    result = await FairHousingEvaluator().evaluate("case_8", "This listing is best suited for able-bodied residents.")
    details = assert_fh_details(result)
    assert result.verdict == "fail"
    assert any(m.category == "disability" for m in details.tripwire_matches)


async def test_mature_adults_flagged_for_human_review():
    """Lawful only under HUD's specific senior-housing exemption criteria,
    which text alone can't verify -- must flag, never silently pass."""
    result = await FairHousingEvaluator().evaluate("case_9", "A wonderful community for active or mature adults.")
    assert result.verdict == "fail"


# ── The church vs. house-of-worship distinction ─────────────────────────


async def test_generic_houses_of_worship_nearby_not_flagged():
    """HUD explicitly lists 'houses of worship nearby' as an ACCEPTABLE
    phrase, distinct from naming a specific worship-institution type."""
    result = await FairHousingEvaluator().evaluate(
        "case_10", "Quiet residential area with houses of worship nearby."
    )
    details = assert_fh_details(result)
    assert result.verdict == "pass"
    assert details.tripwire_matches == []


# ── False-positive guards: HUD's own "acceptable" examples ─────────────


async def test_master_bedroom_not_flagged():
    result = await FairHousingEvaluator().evaluate(
        "case_11", "The master bedroom features a walk-in closet and en-suite bath."
    )
    assert result.verdict == "pass"


async def test_family_room_not_flagged():
    """Bare-word matching on 'family' would falsely flag this. Confirms
    the rule set is phrase-level, not single-token."""
    result = await FairHousingEvaluator().evaluate("case_12", "A spacious family room opens onto the covered patio.")
    assert result.verdict == "pass"


async def test_quiet_residential_area_not_flagged():
    result = await FairHousingEvaluator().evaluate("case_13", "Located in a quiet residential area close to parks.")
    assert result.verdict == "pass"


async def test_parks_nearby_not_flagged():
    result = await FairHousingEvaluator().evaluate("case_14", "Enjoy parks nearby and easy access to the highway.")
    assert result.verdict == "pass"


async def test_gated_not_flagged():
    result = await FairHousingEvaluator().evaluate("case_15", "This gated community offers extra peace of mind.")
    assert result.verdict == "pass"


async def test_rare_find_and_desirable_neighborhood_not_flagged():
    result = await FairHousingEvaluator().evaluate(
        "case_16", "A rare find in one of the area's most desirable neighborhoods."
    )
    assert result.verdict == "pass"


async def test_safety_features_not_flagged():
    """Guards against a naive 'safe' tripwire catching legitimate safety
    feature descriptions -- 'safe' is deliberately not in the rule set at
    all for exactly this reason."""
    result = await FairHousingEvaluator().evaluate(
        "case_17", "Equipped with modern safety features including a smoke detector system."
    )
    assert result.verdict == "pass"


async def test_exclusive_listing_not_flagged():
    """'Exclusive' is deliberately excluded from the regex rule set --
    it's common, benign marketing language most of the time. See module
    docstring for why it's deferred to the judge layer instead."""
    result = await FairHousingEvaluator().evaluate("case_18", "An exclusive listing in a sought-after enclave.")
    assert result.verdict == "pass"


# ── No-judge-available behavior ─────────────────────────────────────────


async def test_no_judge_available_is_explicit_not_silent():
    """When the regex layer finds nothing and no judge_client is passed,
    the evaluator returns verdict=pass -- but this test exists to make
    that behavior explicit and intentional, not an accidental side effect
    of judge_client defaulting to None. If this test starts failing after
    a judge is wired in, that's expected -- update it to reflect the new
    escalation path rather than deleting it."""
    result = await FairHousingEvaluator().evaluate("case_19", "A lovely home with a spacious backyard.")
    details = assert_fh_details(result)
    assert result.verdict == "pass"
    assert details.judge_escalation is None
    assert result.evaluator_type == "deterministic"
    assert result.model_used is None
    assert result.cost_usd is None


# ── Judge escalation path (mock judge, no real API call) ────────────────


async def test_judge_escalation_high_risk_fails():
    judge = MockJudgeClient(risk_level=0.85, rationale="Implies exclusion via framing.")
    result = await FairHousingEvaluator().evaluate("case_20", "Some subtly exclusionary text.", judge_client=judge)
    details = assert_fh_details(result)

    assert result.verdict == "fail"
    assert details.final_risk == "high"
    assert result.evaluator_type == "llm_judge"
    assert result.score == 0.85
    assert result.model_used == "mock-judge-v1"
    assert result.cost_usd == 0.0001
    assert result.latency_ms == 42
    assert details.judge_escalation is not None
    assert details.judge_escalation.rationale == "Implies exclusion via framing."


async def test_judge_escalation_low_risk_flags():
    judge = MockJudgeClient(risk_level=0.45)
    result = await FairHousingEvaluator().evaluate("case_21", "Borderline text.", judge_client=judge)
    details = assert_fh_details(result)

    assert result.verdict == "flag"
    assert details.final_risk == "low"


async def test_judge_escalation_no_risk_passes():
    judge = MockJudgeClient(risk_level=0.05)
    result = await FairHousingEvaluator().evaluate("case_22", "Clearly neutral text.", judge_client=judge)
    details = assert_fh_details(result)

    assert result.verdict == "pass"
    assert details.final_risk == "none"
    assert result.evaluator_type == "llm_judge"  # judge WAS called, even though it found nothing


async def test_judge_not_called_when_regex_already_resolved():
    """If the regex layer finds a match, the judge must never be called --
    that's the whole point of the two-layer cost structure. A judge that
    raises AssertionError proves it was never invoked."""

    class ExplodingJudge(JudgeClient):
        model_name = "should-never-be-called"

        async def score_quality(self, *a, **kw):
            raise AssertionError("quality judge should not be called by fair_housing tests")

        async def score_fair_housing_risk(self, text: str) -> JudgeCallResult:
            raise AssertionError("judge should never be called when regex already found a match")

    result = await FairHousingEvaluator().evaluate(
        "case_23", "No children allowed.", judge_client=ExplodingJudge()
    )
    assert result.verdict == "fail"
    assert result.evaluator_type == "deterministic"