"""
Gemini judge implementation.

Uses `google-genai` (the unified SDK), not the deprecated
`google-generativeai` package -- see judges/pricing.py's module docstring
for the same deprecation note. Install with `pip install google-genai`.

IMPORTANT -- not verified against the live API: this sandbox has no
network access, so this class is implemented against documented
google-genai usage patterns but has NOT been exercised against a real
Gemini response. Specifically unverified:

  - `client.aio.models.generate_content(...)` as the async call shape
  - `response.usage_metadata.prompt_token_count` /
    `.candidates_token_count` as the token-usage attribute names

Smoke-test this against a real GEMINI_API_KEY before trusting it in the
harness. If the SDK's actual response shape differs from what's assumed
below, this is exactly the kind of thing the Third-Party Debugging
Protocol exists for: add logging of the raw response object first, form
a hypothesis from what's actually returned, then fix -- don't guess twice.
"""

from __future__ import annotations

import time
from pathlib import Path

from google import genai

from evaluators.base import JudgeDetails
from judges.base import JudgeCallResult, JudgeClient, JudgeParseError, load_prompt_and_hash, parse_judge_json
from judges.pricing import DEFAULT_JUDGE_MODEL, compute_cost

PROMPTS_DIR = Path(__file__).parent / "prompts"


class GeminiJudge(JudgeClient):
    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL, api_key: str | None = None):
        self.model_name = model_name
        # genai.Client() reads GEMINI_API_KEY from the environment if
        # api_key isn't passed explicitly.
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

        self._quality_prompt, self._quality_prompt_version = load_prompt_and_hash(
            PROMPTS_DIR / "quality_judge.txt"
        )
        self._fh_prompt, self._fh_prompt_version = load_prompt_and_hash(
            PROMPTS_DIR / "fair_housing_judge.txt"
        )

    async def score_quality(
        self, mls_summary: str, headline: str, property_context: dict
    ) -> JudgeCallResult:
        prompt = self._quality_prompt.format(
            property_context=property_context,
            headline=headline,
            mls_summary=mls_summary,
        )
        return await self._call(prompt, self._quality_prompt_version)

    async def score_fair_housing_risk(self, text: str) -> JudgeCallResult:
        prompt = self._fh_prompt.format(text=text)
        return await self._call(prompt, self._fh_prompt_version)

    async def _call(self, prompt: str, prompt_version: str) -> JudgeCallResult:
        start = time.monotonic()
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        latency_ms = int((time.monotonic() - start) * 1000)

        raw_text = response.text
        if raw_text is None:
            # Real failure mode, not hypothetical: safety-filtered or
            # empty-candidate responses come back with text=None. Fail
            # loudly here with a clear cause, rather than let None reach
            # parse_judge_json and produce a confusing error two calls
            # deeper -- same "never silently swallow a judge failure"
            # principle as parse_judge_json itself.
            raise JudgeParseError(
                "Judge response had no text content -- likely blocked by "
                "safety filters or returned an empty candidate.",
                raw_response="",
            )
        parsed = parse_judge_json(raw_text)  # raises JudgeParseError on malformed output

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        cost_usd = compute_cost(self.model_name, input_tokens, output_tokens)

        details = JudgeDetails(
            rationale=parsed.get("rationale", ""),
            criteria_scores=parsed.get("criteria_scores", {}),
            raw_judge_response=raw_text,
            prompt_version=prompt_version,
        )
        return JudgeCallResult(
            details=details,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            model_used=self.model_name,
        )