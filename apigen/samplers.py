"""Sampling system for dataset diversity.

API Sampler, Seed QA Sampler, and Prompt Sampler. The number of APIs and seed
examples per iteration is randomly chosen within a configured range to avoid
repetitive patterns (the paper's Sampling Diversity).
"""

from __future__ import annotations

import random
from typing import get_args

from .schemas import APIDef, QAPair, QueryStyle

ALL_STYLES: tuple[QueryStyle, ...] = get_args(QueryStyle)


class ApiSampler:
    def __init__(self, apis: list[APIDef], count_range: tuple[int, int], rng: random.Random | None = None):
        self._apis = apis
        self._range = count_range
        self._rng = rng or random.Random()

    def sample(self) -> list[APIDef]:
        lo, hi = self._range
        hi = min(hi, len(self._apis))
        lo = min(lo, hi)
        k = self._rng.randint(lo, hi) if hi >= lo else 0
        return self._rng.sample(self._apis, k)


class SeedQASampler:
    def __init__(self, seeds: list[QAPair], count_range: tuple[int, int], rng: random.Random | None = None):
        self._seeds = list(seeds)
        self._range = count_range
        self._rng = rng or random.Random()

    def sample(self, style: QueryStyle | None = None) -> list[QAPair]:
        pool = self._seeds
        if style is not None:
            styled = [s for s in self._seeds if s.style == style]
            # Fall back to the full pool if there aren't enough style matches.
            pool = styled if styled else self._seeds
        lo, hi = self._range
        hi = min(hi, len(pool))
        lo = min(lo, hi)
        k = self._rng.randint(lo, hi) if hi >= lo else 0
        return self._rng.sample(pool, k)

    def add(self, pair: QAPair) -> None:
        """Feedback loop: add a verified pair back into the seed pool."""
        self._seeds.append(pair)


class PromptSampler:
    def __init__(self, styles: tuple[QueryStyle, ...] = ALL_STYLES, rng: random.Random | None = None):
        self._styles = styles
        self._rng = rng or random.Random()

    def sample(self) -> QueryStyle:
        return self._rng.choice(self._styles)
