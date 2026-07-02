"""
Swappable judge interface.

Async, not sync -- judge calls are I/O-bound API requests, and async lets
the harness run multiple golden-dataset cases concurrently instead of
serially once the dataset grows past a handful of cases. This is the one
interface both quality_judge.py and fair_housing.py's escalation path
depend on, so getting the shape right here matters more than in most
files -- every judge implementation (Gemini now, Claude/GPT later) has to
satisfy the same contract.

Design decisions:

- `prompt_version` is a content hash, computed by `load_prompt_and_hash`
  at prompt-load time, not a hand-maintained string. See the discussion
  in evaluators/base.py's JudgeDetails -- this is the concrete mechanism
  that field relies on.

- `parse_judge_json` raises on malformed output rather than returning a
  default/empty result. A judge call that can't be parsed is NOT the same
  as a judge call that found no issues -- silently treating them the same
  would let a parsing bug masquerade as a clean compliance pass. Callers
  must handle the exception explicitly (e.g. record it as an eval error,
  not a pass).
"""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple, Optional

from evaluators.base import JudgeDetails


class JudgeCallResult(NamedTuple):
    """Everything a caller needs to build a complete EvalResult from one
    judge call. JudgeDetails alone isn't enough -- latency and cost live
    on EvalResult, not JudgeDetails, but only the judge implementation
    actually has the token-usage numbers from the raw API response. This
    wrapper is how that information gets out without EvalResult-shaped
    fields leaking into JudgeDetails, which is meant to be judge-agnostic."""

    details: JudgeDetails
    latency_ms: int
    cost_usd: Optional[float]
    model_used: str


class JudgeParseError(Exception):
    """Raised when a judge's response can't be parsed into the expected
    structure. Carries the raw response so the caller can log/inspect it --
    per the Third-Party Debugging Protocol, never theorize about what a
    model returned, always have the actual text on hand."""

    def __init__(self, message: str, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


def load_prompt_and_hash(path: Path) -> tuple[str, str]:
    """Reads a prompt file and returns (prompt_text, version_hash).
    version_hash is sha256(prompt_text)[:8] -- content-addressed, so it's
    structurally impossible for a JudgeDetails.prompt_version to claim a
    prompt version that doesn't match the text that actually ran."""
    prompt_text = path.read_text()
    version_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:8]
    return prompt_text, version_hash


def parse_judge_json(raw_response: str) -> dict:
    """Parses a judge's raw text response into a dict. Handles the common
    failure mode of a model wrapping JSON in markdown code fences despite
    being asked not to. Raises JudgeParseError (carrying the raw text) on
    anything else -- never returns a partial or default result, because a
    parse failure must never be indistinguishable from a clean judge
    verdict downstream."""
    stripped = raw_response.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else stripped

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise JudgeParseError(
            f"Judge response could not be parsed as JSON: {e}", raw_response
        ) from e


class JudgeClient(ABC):
    """Every judge implementation (Gemini, Claude, GPT) satisfies this
    contract. Callers (quality_judge.py, fair_housing.py's escalation
    path) depend only on this interface, never on a concrete judge class --
    that's what makes "evaluate one model with another" a config swap
    instead of a code change."""

    model_name: str

    @abstractmethod
    async def score_quality(
        self, mls_summary: str, headline: str, property_context: dict
    ) -> JudgeCallResult:
        """Scores unstructured hallucination, completeness, and tone for
        listing copy that fact_diff.py's deterministic checks can't cover.
        property_context is the case's `input` block (PropertyDetails-
        shaped dict) so the judge can ground its assessment in what was
        actually provided."""
        raise NotImplementedError

    @abstractmethod
    async def score_fair_housing_risk(self, text: str) -> JudgeCallResult:
        """Escalation path for fair_housing.py -- only called when the
        regex tripwire layer finds nothing. The returned JudgeDetails must
        include a 'fair_housing_risk_level' key in criteria_scores
        (0.0-1.0) -- FairHousingEvaluator reads that specific key to set
        final_risk."""
        raise NotImplementedError