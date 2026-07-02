"""
Deterministic fact-diff evaluator.

Checks whether generated copy (mls_summary + headline) states the facts it
should state, and does NOT state facts it has no basis for. This is
deliberately NOT an LLM call -- see the design discussion in the harness
README. Hard facts (sqft, beds, baths, named features) can be verified with
string matching; spending a judge call on them would just reintroduce
variance on a problem that doesn't need it.

Matching strategy (`normalize_and_match`):

- Strips comma formatting from numbers ("1,850" == "1850").
- Matches digit and word forms of numbers 1-20 ("3" == "three").
- Matches real-estate unit synonyms ("bed" == "bedroom" == "beds" == "BR").
- Falls through to a plain normalized substring match for facts with no
  leading number (e.g. "granite countertops").

Deliberately NOT fuzzy beyond this. No semantic matching, no embeddings.
If a fact doesn't appear in any of these recognizable forms, that's treated
as a real miss, not a matching bug. Known limitation: compound facts like
"2.5 bath" won't match copy phrased as "2 full baths and 1 half bath" --
see test_fact_diff.py::test_half_bath_known_limitation. Fixing that
properly needs a bathroom-specific parser, deferred rather than special-
cased into this general-purpose matcher.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from evaluators.base import EvalResult, FactDiffDetails

NUMBER_WORDS: dict[int, str] = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
    11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
    15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
    19: "nineteen", 20: "twenty",
}

# Canonical unit key -> accepted surface forms in generated copy.
UNIT_SYNONYMS: dict[str, list[str]] = {
    "bed": ["bed", "beds", "bedroom", "bedrooms", "br"],
    "bath": ["bath", "baths", "bathroom", "bathrooms", "ba"],
    "sqft": [
        "sqft", "sq ft", "sq. ft.", "square feet", "square footage",
        "square-foot", "square foot",
    ],
    "garage": ["garage", "car garage"],
    "story": ["story", "stories", "storey", "storeys", "floor"],
}

# fact-string unit token -> canonical key, used to look up UNIT_SYNONYMS
_UNIT_ALIAS_TO_KEY: dict[str, str] = {}
for _key, _synonyms in UNIT_SYNONYMS.items():
    for _syn in _synonyms:
        _UNIT_ALIAS_TO_KEY[_syn] = _key


def _strip_number_commas(text: str) -> str:
    """'1,850' -> '1850'. Repeats to handle multi-comma numbers."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"(\d),(\d{3})", r"\1\2", text)
    return text


def _normalize(text: str) -> str:
    text = text.lower()
    text = _strip_number_commas(text)
    return text


WORD_TO_NUMBER: dict[str, int] = {word: n for n, word in NUMBER_WORDS.items()}
_NUMBER_WORD_PATTERN = "|".join(NUMBER_WORDS.values())


def _number_variants(token: str) -> list[str]:
    """Given a number token from a fact string -- digit ('3') or word
    ('three') -- return every recognized form so matching works regardless
    of which form the fact or the generated copy happens to use."""
    variants = [token]
    if re.match(r"^\d+(?:\.\d+)?$", token):
        try:
            n_int = int(float(token))
            if token == str(n_int) and n_int in NUMBER_WORDS:
                variants.append(NUMBER_WORDS[n_int])
        except ValueError:
            pass
    elif token in WORD_TO_NUMBER:
        variants.append(str(WORD_TO_NUMBER[token]))
    return variants


def _unit_key(unit_phrase: str) -> Optional[str]:
    unit_phrase = unit_phrase.strip()
    if not unit_phrase:
        return None
    return _UNIT_ALIAS_TO_KEY.get(unit_phrase)


def normalize_and_match(fact: str, generated_text: str) -> bool:
    """Return True if `fact` is recognizably present in `generated_text`."""
    fact_norm = _normalize(fact.strip())
    text_norm = _normalize(generated_text)

    m = re.match(rf"^(\d+(?:\.\d+)?|{_NUMBER_WORD_PATTERN})\s*(.*)$", fact_norm)
    if not m:
        # Plain phrase fact, e.g. "granite countertops" -- substring match.
        return fact_norm in text_norm

    number_str, unit_phrase = m.groups()
    unit_phrase = unit_phrase.strip()
    number_candidates = _number_variants(number_str)

    if not unit_phrase:
        # Bare number fact, e.g. "1850" -- match the number on its own,
        # with a word boundary so "1850" doesn't match inside "18505".
        return any(
            re.search(rf"\b{re.escape(n)}\b", text_norm) for n in number_candidates
        )

    unit_key = _unit_key(unit_phrase)
    unit_candidates = UNIT_SYNONYMS.get(unit_key, [unit_phrase]) if unit_key else [unit_phrase]

    for n in number_candidates:
        for u in unit_candidates:
            # Allow a space, hyphen, or nothing between number and unit:
            # "3 bed", "3-bedroom", "three bedrooms", "3BR" (no separator).
            # (?<!\d) before the number stops "13" from matching fact "3".
            # \b after the unit stops "bedroomy" from matching "bedroom".
            # No \b required between number and unit -- digits and letters
            # are both \w chars, so "3br" has no boundary between "3" and
            # "b" for \b to anchor on.
            pattern = rf"(?<!\d){re.escape(n)}[\s-]*{re.escape(u)}\b"
            if re.search(pattern, text_norm):
                return True
    return False


class FactDiffEvaluator:
    """Runs normalize_and_match across a golden case's expected facts."""

    evaluator_name = "fact_diff"
    evaluator_type = "deterministic"

    def evaluate(self, case: dict, generated_text: str) -> EvalResult:
        """
        `case` is a golden-dataset case dict (see data/golden_dataset.json).
        `generated_text` is the copy under test -- mls_summary + headline,
        concatenated, so a fact stated in either one counts as present.
        """
        expected = case["expected"]["mls_summary"]
        must_include: list[str] = expected.get("must_include_facts", [])
        must_not_include: list[str] = expected.get("must_not_include", [])

        facts_present: list[str] = []
        facts_missing: list[str] = []
        for fact in must_include:
            if normalize_and_match(fact, generated_text):
                facts_present.append(fact)
            else:
                facts_missing.append(fact)

        facts_fabricated: list[str] = [
            fact for fact in must_not_include if normalize_and_match(fact, generated_text)
        ]

        if facts_fabricated:
            verdict = "fail"
        elif facts_missing:
            verdict = "flag"
        else:
            verdict = "pass"

        details = FactDiffDetails(
            facts_checked=must_include + must_not_include,
            facts_present=facts_present,
            facts_missing=facts_missing,
            facts_fabricated=facts_fabricated,
        )

        return EvalResult(
            evaluator_name=self.evaluator_name,
            evaluator_type=self.evaluator_type,
            case_id=case["case_id"],
            verdict=verdict,
            score=None,
            details=details,
            latency_ms=0,  # deterministic -- negligible, but recorded honestly rather than omitted
            cost_usd=None,
            model_used=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
