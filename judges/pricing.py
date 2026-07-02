"""
Judge model pricing table.

Cost tracking (EvalResult.cost_usd) depends entirely on this table being
current. If it goes stale, cost_usd silently reports wrong numbers rather
than erroring -- there's no way for the harness to know pricing changed.
`LAST_VERIFIED` exists so that staleness is at least visible, not silent.

Rates are USD per 1,000,000 tokens, input/output, standard interactive
pricing (not batch/flex, not cached-input rates). Source: ai.google.dev
pricing page, verified as of the date below. Gemini 2.0 Flash and Flash-
Lite were deprecated and shut down June 1, 2026 -- do not default to
those model names even though they may still appear in older code
examples or training data.
"""

from __future__ import annotations

from datetime import date
from typing import NamedTuple, Optional

LAST_VERIFIED = date(2026, 7, 2)


class ModelPricing(NamedTuple):
    input_per_million: float
    output_per_million: float


PRICING_TABLE: dict[str, ModelPricing] = {
    # Gemini -- current generation as of LAST_VERIFIED
    "gemini-2.5-flash-lite": ModelPricing(0.10, 0.40),
    "gemini-2.5-flash": ModelPricing(0.30, 2.50),
    "gemini-3.1-flash-lite": ModelPricing(0.25, 1.50),
    "gemini-3-flash": ModelPricing(0.50, 3.00),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00),  # base tier, <=200k context
    # Claude / GPT -- placeholders for the independent-judge swap-in.
    # Verify against anthropic.com/pricing and openai.com/pricing before
    # relying on these for real cost tracking -- not verified as part of
    # this build, unlike the Gemini rates above.
}

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Returns None (not 0.0) if the model isn't in the table -- an
    unknown model's cost is unknown, not free. Callers should treat None
    as 'cost tracking unavailable for this call', matching the same
    Optional convention used throughout EvalResult."""
    pricing = PRICING_TABLE.get(model)
    if pricing is None:
        return None
    return (
        input_tokens / 1_000_000 * pricing.input_per_million
        + output_tokens / 1_000_000 * pricing.output_per_million
    )