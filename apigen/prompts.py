"""Prompt templates for the four query styles.

Each builds a chat prompt that steers the generator LLM to emit a JSON array of
{query, answers} objects (the batching trick: many pairs per inference).
"""

from __future__ import annotations

import json

from .schemas import APIDef, QAPair, QueryStyle

STYLE_INSTRUCTIONS: dict[QueryStyle, str] = {
    "simple": (
        "Generate queries that each require exactly ONE function call to satisfy. "
        "Use a single API per query."
    ),
    "multiple": (
        "You are given several APIs. Generate queries where each query is satisfied by exactly ONE "
        "function call, but the correct API must be chosen from among the several provided."
    ),
    "parallel": (
        "Generate queries that each require MULTIPLE function calls (to the same or different APIs) "
        "executed together to be fully satisfied. Each query's 'answers' list must contain more than one call."
    ),
    "parallel_multiple": (
        "You are given several APIs. Generate queries that each require MULTIPLE function calls drawn "
        "from the different provided APIs, all needed together to satisfy the request."
    ),
}

SYSTEM_PROMPT = (
    "You are a data generator for a user-car voice-assistant function-calling dataset. "
    "Given a set of car-control APIs, you produce realistic natural-language driver queries paired "
    "with the exact function calls that fulfill them. You only use the provided APIs and their "
    "declared parameters. You never invent functions or arguments."
)

OUTPUT_CONTRACT = """\
Respond with ONLY a JSON array (no prose, no markdown fences). Each element must be an object:
  {"query": "<natural language request>", "answers": [{"name": "<api_name>", "arguments": {"<arg>": <value>}}]}
Rules:
- Use only the API names and parameter names listed above.
- Include every required parameter; use values within any stated valid range.
- "arguments" values must be concrete (numbers, strings, booleans), never placeholders.
"""


def _format_apis(apis: list[APIDef]) -> str:
    return json.dumps([api.model_dump() for api in apis], indent=2)


def _format_seeds(seeds: list[QAPair]) -> str:
    if not seeds:
        return "(none)"
    return json.dumps(
        [{"query": s.query, "answers": [a.model_dump() for a in s.answers]} for s in seeds],
        indent=2,
    )


def build_messages(
    style: QueryStyle,
    apis: list[APIDef],
    seeds: list[QAPair],
    num_pairs: int,
) -> list[dict[str, str]]:
    """Build the chat messages for one generation request."""
    user = f"""\
## Available APIs
{_format_apis(apis)}

## Example query-answer pairs (for style reference)
{_format_seeds(seeds)}

## Task
{STYLE_INSTRUCTIONS[style]}
Generate {num_pairs} diverse query-answer pair(s) in this style.

## Output format
{OUTPUT_CONTRACT}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
