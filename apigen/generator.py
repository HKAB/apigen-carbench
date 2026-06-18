"""Query-Answer Generator: calls the generator LLM and parses batched output."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from .llm_client import LLMClient
from .prompts import build_messages
from .schemas import APIDef, GeneratorOutput, Persona, QAPair, QueryStyle

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json_array(text: str) -> str:
    """Best-effort extraction of a JSON array from an LLM response.

    Handles raw arrays, ```json fenced blocks, and surrounding prose.
    """
    text = text.strip()
    fence = _FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    # If there is leading/trailing prose, slice to the outermost brackets.
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def parse_generator_output(text: str) -> list[GeneratorOutput]:
    """Parse the raw LLM text into GeneratorOutput objects.

    Malformed items are skipped (the format checker is the authoritative gate;
    here we just salvage what parses). A completely unparseable response yields [].
    """
    raw = _extract_json_array(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    out: list[GeneratorOutput] = []
    for item in data:
        try:
            out.append(GeneratorOutput.model_validate(item))
        except ValidationError:
            continue
    return out


class QueryAnswerGenerator:
    def __init__(self, client: LLMClient, model: str, temperature: float = 0.7):
        self._client = client
        self._model = model
        self._temperature = temperature

    async def generate(
        self,
        apis: list[APIDef],
        seeds: list[QAPair],
        style: QueryStyle,
        num_pairs: int = 3,
        persona: Persona | None = None,
        real_examples: list[QAPair] | None = None,
    ) -> list[GeneratorOutput]:
        messages = build_messages(
            style, apis, seeds, num_pairs, persona=persona, real_examples=real_examples
        )
        text = await self._client.chat(self._model, messages, self._temperature)
        return parse_generator_output(text)
