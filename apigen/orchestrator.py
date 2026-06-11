"""Async orchestrator: sample -> generate -> verify -> feedback.

Bounded-concurrency design: each generation iteration runs under a shared
semaphore; generated pairs are verified concurrently. Verified pairs are
written to the output sink and fed back into the seed pool. Per-stage filter
counts are tracked, mirroring the paper's statistics table.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from .feedback import FeedbackLoop
from .generator import QueryAnswerGenerator
from .samplers import ApiSampler, PersonaSampler, PromptSampler, SeedQASampler
from .schemas import GeneratorOutput, QAPair
from .verification.pipeline import VerificationPipeline


@dataclass
class Stats:
    generated: int = 0
    verified: int = 0
    fail_format: int = 0
    fail_execution: int = 0
    fail_semantic: int = 0
    _by_stage: dict[str, str] = field(default_factory=dict)

    def record(self, result_passed: bool, failed_stage: str | None) -> None:
        if result_passed:
            self.verified += 1
        elif failed_stage == "format":
            self.fail_format += 1
        elif failed_stage == "execution":
            self.fail_execution += 1
        elif failed_stage == "semantic":
            self.fail_semantic += 1

    @property
    def pass_rate(self) -> float:
        return (self.verified / self.generated) if self.generated else 0.0

    def summary(self) -> str:
        return (
            f"generated={self.generated} verified={self.verified} "
            f"fail_format={self.fail_format} fail_execution={self.fail_execution} "
            f"fail_semantic={self.fail_semantic} pass_rate={self.pass_rate:.2%}"
        )


class Orchestrator:
    def __init__(
        self,
        generator: QueryAnswerGenerator,
        pipeline: VerificationPipeline,
        api_sampler: ApiSampler,
        seed_sampler: SeedQASampler,
        prompt_sampler: PromptSampler,
        persona_sampler: PersonaSampler,
        feedback: FeedbackLoop,
        concurrency: int = 8,
        pairs_per_batch: int = 3,
    ):
        self._generator = generator
        self._pipeline = pipeline
        self._api_sampler = api_sampler
        self._seed_sampler = seed_sampler
        self._prompt_sampler = prompt_sampler
        self._feedback = feedback
        self._concurrency = max(1, concurrency)
        self._sem = asyncio.Semaphore(self._concurrency)
        self._pairs_per_batch = pairs_per_batch
        self._persona_sampler = persona_sampler  # None → no persona injection
        self.stats = Stats()
        self._lock = asyncio.Lock()

    async def _run_one_batch(self) -> list[QAPair]:
        """Generate one batch and verify each pair; return the verified pairs."""
        async with self._sem:
            style = self._prompt_sampler.sample()
            apis = self._api_sampler.sample()
            if not apis:
                return []
            seeds = self._seed_sampler.sample(style)
            # Sample a fresh persona per batch for maximum query diversity.
            persona = self._persona_sampler.sample() if self._persona_sampler else None
            items = await self._generator.generate(
                apis, seeds, style, self._pairs_per_batch, persona=persona
            )

        accepted: list[QAPair] = []
        for item in items:
            result = await self._pipeline.verify(item)
            async with self._lock:
                self.stats.generated += 1
                self.stats.record(result.passed, result.failed_stage)
                if result.passed:
                    accepted.append(self._feedback.accept(item, style))
        return accepted

    async def run(self, target: int, max_waves: int | None = None) -> list[QAPair]:
        """Run until at least `target` verified pairs are collected.

        Launches batches in waves sized to the concurrency limit. To avoid an
        unbounded loop when generation never produces verifiable data, the run
        stops after `max_waves` waves (default: enough waves to plausibly reach
        the target, with headroom).
        """
        if max_waves is None:
            per_wave = self._concurrency * self._pairs_per_batch
            max_waves = max(4, (target // max(1, per_wave) + 1) * 4)

        verified: list[QAPair] = []
        waves = 0
        while len(verified) < target and waves < max_waves:
            results = await asyncio.gather(
                *(self._run_one_batch() for _ in range(self._concurrency))
            )
            for batch in results:
                verified.extend(batch)
            waves += 1
        return verified[:target]
