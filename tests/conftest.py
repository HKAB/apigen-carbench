"""Shared test fixtures: a scriptable fake LLM client and a loaded API library."""

from __future__ import annotations

from pathlib import Path

import pytest

from apigen.api_library import ApiLibrary

ROOT = Path(__file__).resolve().parent.parent
APIS_PATH = ROOT / "data" / "apis" / "car_apis.json"
SEED_PATH = ROOT / "data" / "seed" / "seed_qa.json"


class FakeLLMClient:
    """LLMClient stub.

    `responses` maps a model name to a list of canned response strings, served
    in order per model. If a model has a single response, it is reused. Records
    all calls for assertions.
    """

    def __init__(self, responses: dict[str, list[str]] | None = None):
        self._responses = {k: list(v) for k, v in (responses or {}).items()}
        self.calls: list[dict] = []

    async def chat(self, model, messages, temperature=0.7) -> str:
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        queue = self._responses.get(model)
        if not queue:
            return "[]"
        if len(queue) == 1:
            return queue[0]
        return queue.pop(0)


@pytest.fixture
def library() -> ApiLibrary:
    return ApiLibrary.from_json(APIS_PATH)


@pytest.fixture
def apis_path() -> Path:
    return APIS_PATH


@pytest.fixture
def seed_path() -> Path:
    return SEED_PATH
