"""End-to-end: orchestrator with a fake LLM, no GPU/network."""

import json
import random

import pytest

from apigen.backends import PythonExecutionBackend
from apigen.feedback import FeedbackLoop
from apigen.generator import QueryAnswerGenerator
from apigen.orchestrator import Orchestrator
from apigen.samplers import ApiSampler, PromptSampler, SeedQASampler
from apigen.schemas import QAPair
from apigen.verification import (
    ExecutionChecker,
    FormatChecker,
    SemanticChecker,
    VerificationPipeline,
)
from tests.conftest import FakeLLMClient

GEN = "gen-model"
SEM = "sem-model"

# A batch with one good pair and one hallucinated-function pair (fails format).
GEN_BATCH = json.dumps(
    [
        {"query": "set the volume to 12", "answers": [{"name": "set_volume", "arguments": {"level": 12}}]},
        {"query": "do the impossible", "answers": [{"name": "teleport", "arguments": {}}]},
    ]
)


def _build_orchestrator(library, seeds, client, concurrency=2):
    rng = random.Random(0)
    api_sampler = ApiSampler(library.all(), (1, 4), rng)
    seed_sampler = SeedQASampler(seeds, (0, 2), rng)
    prompt_sampler = PromptSampler(rng=rng)
    generator = QueryAnswerGenerator(client, GEN, 0.7)
    pipeline = VerificationPipeline(
        FormatChecker(library),
        ExecutionChecker(PythonExecutionBackend()),
        SemanticChecker(client, SEM),
        library,
    )
    feedback = FeedbackLoop(seed_sampler)
    return Orchestrator(
        generator, pipeline, api_sampler, seed_sampler, prompt_sampler, feedback,
        concurrency=concurrency, pairs_per_batch=2,
    ), seed_sampler


@pytest.fixture
def seeds(seed_path):
    return [QAPair.model_validate(x) for x in json.loads(seed_path.read_text())]


async def test_good_passes_bad_filtered(library, seeds):
    client = FakeLLMClient({GEN: [GEN_BATCH], SEM: ['{"pass": true, "reason": "ok"}']})
    orch, seed_sampler = _build_orchestrator(library, seeds, client)

    verified = await orch.run(target=1)

    # The good pair is verified; the hallucinated one is filtered at format.
    assert len(verified) >= 1
    assert all(p.answers[0].name == "set_volume" for p in verified)
    assert orch.stats.verified >= 1
    assert orch.stats.fail_format >= 1


async def test_semantic_rejection_filters(library, seeds):
    client = FakeLLMClient({GEN: [GEN_BATCH], SEM: ['{"pass": false, "reason": "nope"}']})
    orch, _ = _build_orchestrator(library, seeds, client)

    # Semantic checker rejects everything -> nothing verified, capped by max_waves.
    verified = await orch.run(target=1, max_waves=2)

    assert verified == []
    assert orch.stats.fail_semantic >= 1
    assert orch.stats.verified == 0


async def test_feedback_loop_grows_seed_pool(library, seeds):
    client = FakeLLMClient({GEN: [GEN_BATCH], SEM: ['{"pass": true, "reason": "ok"}']})
    orch, seed_sampler = _build_orchestrator(library, seeds, client)
    before = len(seed_sampler._seeds)

    await orch.run(target=1)

    assert len(seed_sampler._seeds) > before
