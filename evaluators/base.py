"""
Core evaluation result schema.

Design decisions (see README for full rationale):

- One outer envelope (`EvalResult`) shared by every evaluator, with a typed
  `details` payload that varies by evaluator family. This is a discriminated
  union: the report/trace/regression code only ever needs to understand the
  envelope, never the three different detail shapes underneath it.

- `verdict` is a separate field from `score`. Fact-diff has no meaningful
  continuous score (a fact is present or fabricated, full stop); the judge
  does. Keeping them structurally separate makes it impossible to
  accidentally average a deterministic binary with a fuzzy judge score --
  a real trap if they lived in the same field.

- `cost_usd` / `model_used` are Optional, not defaulted to 0.0 / "none".
  0.0 is ambiguous (genuinely free, or did we fail to record it?). None is
  unambiguous: this evaluator doesn't call a model.

- `prompt_version` on JudgeDetails is a hash of the actual prompt text,
  computed at load time -- not a hand-maintained string. See
  judges/prompt_hash.py. This makes it structurally impossible for the
  version to drift out of sync with the prompt that produced a result.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

EvaluatorType = Literal["deterministic", "llm_judge"]
Verdict = Literal["pass", "fail", "flag"]

# Worst-of ordering used by CaseResult.overall_verdict. A single failing
# evaluator fails the whole case -- we never want an averaged/green dashboard
# hiding a compliance failure behind two happy evaluators.
_VERDICT_SEVERITY: dict[Verdict, int] = {"pass": 0, "flag": 1, "fail": 2}


# ── Deterministic: fact-diff ────────────────────────────────────────────


class FactDiffDetails(BaseModel):
    """Output of the deterministic fact-diff evaluator.

    facts_missing and facts_fabricated are kept as separate lists rather
    than one combined "mismatches" list because they are different failure
    modes with different severity:

    - facts_missing: present in `expected`, not found in generated output.
      A quality miss -- copy is weaker than it could be.
    - facts_fabricated: asserted in generated output, NOT supported by
      `input`. A hallucination -- more serious, and adjacent to Fair
      Housing / liability risk if the fabricated claim is material.
    """

    facts_checked: list[str] = Field(default_factory=list)
    facts_present: list[str] = Field(default_factory=list)
    facts_missing: list[str] = Field(default_factory=list)
    facts_fabricated: list[str] = Field(default_factory=list)


# ── LLM-judge ────────────────────────────────────────────────────────────


class JudgeDetails(BaseModel):
    """Output of any LLM-as-judge call (quality judge, or FH escalation)."""

    rationale: str
    criteria_scores: dict[str, float] = Field(default_factory=dict)
    raw_judge_response: str
    prompt_version: str  # sha256(prompt_text)[:8], computed at load time


# ── Fair Housing ─────────────────────────────────────────────────────────


class TripwireMatch(BaseModel):
    matched_phrase: str
    span_start: int
    span_end: int
    category: str  # e.g. "familial_status", "religion", "national_origin"


class FairHousingDetails(BaseModel):
    tripwire_matches: list[TripwireMatch] = Field(default_factory=list)
    judge_escalation: Optional[JudgeDetails] = None
    final_risk: Literal["none", "low", "high"] = "none"


DetailsPayload = Union[FactDiffDetails, JudgeDetails, FairHousingDetails]


# ── Envelope ─────────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    evaluator_name: str  # "fact_diff" | "quality_judge" | "fair_housing"
    evaluator_type: EvaluatorType
    case_id: str
    verdict: Verdict
    score: Optional[float] = None  # 0-1, only meaningful for llm_judge evaluators
    details: DetailsPayload
    latency_ms: int
    cost_usd: Optional[float] = None
    model_used: Optional[str] = None
    timestamp: str


# ── Roll-ups ─────────────────────────────────────────────────────────────


class CaseResult(BaseModel):
    case_id: str
    category: str
    eval_results: list[EvalResult]
    overall_verdict: Verdict

    @classmethod
    def from_eval_results(
        cls, case_id: str, category: str, eval_results: list[EvalResult]
    ) -> "CaseResult":
        worst = max(
            (r.verdict for r in eval_results),
            key=lambda v: _VERDICT_SEVERITY[v],
            default="pass",
        )
        return cls(
            case_id=case_id,
            category=category,
            eval_results=eval_results,
            overall_verdict=worst,
        )


class RunResult(BaseModel):
    run_id: str
    timestamp: str
    judge_model: str
    case_results: list[CaseResult]
    total_cost_usd: float = 0.0
    total_latency_ms: int = 0
