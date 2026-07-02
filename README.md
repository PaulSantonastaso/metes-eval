# metes-eval

Standalone evaluation and monitoring harness for the [metes.app](https://metes.app) AI listing-generation pipeline.

**Status: active build, in progress.** This is a portfolio project demonstrating LLM evaluation and observability architecture — deterministic fact-checking, regex + LLM-judge hybrid compliance checking, and a swappable judge interface for cross-model evaluation.

Built standalone rather than on top of an off-the-shelf eval platform (e.g. LangSmith) deliberately — see `docs/` for the reasoning once it's written up. Short version: proving the underlying mechanics from primitives, with evaluator logic designed to be portable into a production observability platform later.

## Current state

- `evaluators/base.py` — core `EvalResult` schema (discriminated union across deterministic and LLM-judge evaluator types)
- `evaluators/fact_diff.py` — deterministic fact-checking (hallucination detection for structured claims: beds, baths, sqft, named features). 18 tests passing.
- `evaluators/fair_housing.py` — Fair Housing compliance: regex tripwire layer + LLM-judge escalation hook (judge not yet wired in). 19 tests passing.
- `judges/` — swappable judge interface (Gemini default, Claude/GPT swap-in). In progress.

## Running the tests

```bash
pip install pydantic pytest
pytest tests/ -v
```

## Roadmap

Full architecture writeup, golden dataset expansion, HTML report, regression mode, and hiring-manager-facing documentation are in progress.
