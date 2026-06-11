"""Stage 3: Semantic Checker.

Sends the query, available functions, the chosen calls, and their execution
results to an LLM, which judges whether the results semantically satisfy the
query's intent. Filters out data that executes but does not match the request.
"""

from __future__ import annotations

import json
import re

from ..llm_client import LLMClient
from ..schemas import APIDef, GeneratorOutput, VerifyResult

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

SYSTEM_PROMPT = (
    "You are a strict semantic verifier for a function-calling dataset. Given a user query, the "
    "available APIs, the function calls chosen to answer it, and the real execution results, decide "
    "whether the calls and results FULLY and CORRECTLY satisfy the user's intent. If the query has "
    "multiple requests, every part must be addressed. Be strict: when in doubt, fail."
)

OUTPUT_CONTRACT = (
    'Respond with ONLY a JSON object: {"pass": true|false, "reason": "<short explanation>"}. '
    "No prose, no markdown fences."
)


def _parse_verdict(text: str) -> tuple[bool, str]:
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False, "semantic checker returned unparseable verdict"
    passed = bool(data.get("pass", False))
    reason = str(data.get("reason", ""))
    return passed, reason


class SemanticChecker:
    def __init__(self, client: LLMClient, model: str):
        self._client = client
        self._model = model

    def _build_messages(
        self, item: GeneratorOutput, apis: list[APIDef], execution_results: list
    ) -> list[dict[str, str]]:
        user = f"""\
## User query
{item.query}

## Available APIs
{json.dumps([a.model_dump() for a in apis], indent=2)}

## Chosen function calls
{json.dumps([c.model_dump() for c in item.answers], indent=2)}

## Execution results
{json.dumps(execution_results, indent=2, default=str)}

## Question
Do the chosen function calls and their execution results fully and correctly satisfy the user query?

## Output format
{OUTPUT_CONTRACT}"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    async def check(
        self,
        item: GeneratorOutput,
        apis: list[APIDef],
        execution_results: list,
    ) -> VerifyResult:
        messages = self._build_messages(item, apis, execution_results)
        text = await self._client.chat(self._model, messages, temperature=0.0)
        passed, reason = _parse_verdict(text)
        if passed:
            return VerifyResult.ok(execution_results=execution_results)
        return VerifyResult.fail("semantic", reason or "semantic mismatch")
