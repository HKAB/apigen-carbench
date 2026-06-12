"""Data feedback loop.

Verified high-quality pairs are added back into the seed pool so later
iterations can sample them as few-shot references (improving diversity).
"""

from __future__ import annotations

from .samplers import SeedQASampler
from .schemas import GeneratorOutput, QAPair, QueryStyle


class FeedbackLoop:
    def __init__(self, seed_sampler: SeedQASampler):
        self._seed_sampler = seed_sampler

    def accept(
        self,
        item: GeneratorOutput,
        style: QueryStyle,
        execution_results: list | None = None,
    ) -> QAPair:
        pair = QAPair(
            query=item.query,
            answers=item.answers,
            style=style,
            execution_results=execution_results or [],
        )
        self._seed_sampler.add(pair)
        return pair
