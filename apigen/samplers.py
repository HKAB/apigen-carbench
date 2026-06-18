"""Sampling system for dataset diversity.

API Sampler, Seed QA Sampler, Prompt Sampler, and Persona Sampler. The number
of APIs and seed examples per iteration is randomly chosen within a configured
range to avoid repetitive patterns (the paper's Sampling Diversity).

PersonaSampler draws rows from the nvidia/Nemotron-Personas-Vietnam dataset so
each generation batch is narrated from a distinct Vietnamese user perspective.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import get_args

from .schemas import APIDef, FunctionCall, Persona, QAPair, QueryStyle

ALL_STYLES: tuple[QueryStyle, ...] = get_args(QueryStyle)


class ApiSampler:
    def __init__(self, apis: list[APIDef], count_range: tuple[int, int], rng: random.Random | None = None):
        self._apis = apis
        self._range = count_range
        self._rng = rng or random.Random()

    def sample(self, include: list[str] | None = None) -> list[APIDef]:
        lo, hi = self._range
        hi = min(hi, len(self._apis))
        lo = min(lo, hi)
        k = self._rng.randint(lo, hi) if hi >= lo else 0
        chosen = self._rng.sample(self._apis, k)
        # Force-include named APIs (e.g. a tool that has real few-shot examples).
        if include:
            by_name = {a.name: a for a in self._apis}
            present = {a.name for a in chosen}
            for name in include:
                if name in by_name and name not in present:
                    chosen.append(by_name[name])
                    present.add(name)
        return chosen


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


class PersonaSampler:
    """Sample personas from the nvidia/Nemotron-Personas-Vietnam dataset.

    Typical usage — load once, then call ``sample()`` per generation batch:

    .. code-block:: python

        sampler = PersonaSampler.from_huggingface()          # HuggingFace
        sampler = PersonaSampler.from_jsonl("personas.jsonl")  # local file
        persona = sampler.sample()
    """

    def __init__(self, personas: list[Persona], rng: random.Random | None = None) -> None:
        if not personas:
            raise ValueError("PersonaSampler requires at least one persona.")
        self._personas = personas
        self._rng = rng or random.Random()

    def sample(self) -> Persona:
        """Return one persona drawn uniformly at random."""
        return self._rng.choice(self._personas)

    # ── Loaders ──────────────────────────────────────────────────────────────

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str = "nvidia/Nemotron-Personas-Vietnam",
        split: str = "train",
        max_personas: int | None = None,
        rng: random.Random | None = None,
    ) -> "PersonaSampler":
        """Load personas directly from the HuggingFace Hub.

        Requires the ``datasets`` package::

            pip install datasets
        """
        try:
            from datasets import load_dataset  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Install the 'datasets' package to load personas from HuggingFace: "
                "pip install datasets"
            ) from exc

        ds = load_dataset(dataset_name, split=split)
        if max_personas is not None:
            ds = ds.select(range(min(max_personas, len(ds))))
        personas = [Persona.model_validate(dict(row)) for row in ds]
        return cls(personas, rng=rng)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        max_personas: int | None = None,
        rng: random.Random | None = None,
    ) -> "PersonaSampler":
        """Load personas from a local JSONL file (one JSON object per line)."""
        rows: list[dict] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_personas is not None and len(rows) >= max_personas:
                    break
        personas = [Persona.model_validate(row) for row in rows]
        return cls(personas, rng=rng)

    @classmethod
    def from_list(
        cls,
        records: list[dict],
        rng: random.Random | None = None,
    ) -> "PersonaSampler":
        """Load personas from an already-parsed list of dicts (e.g. from JSON)."""
        personas = [Persona.model_validate(r) for r in records]
        return cls(personas, rng=rng)


class LabelExampleSampler:
    """Real (user-message -> tool) examples, indexed by tool name.

    Source: a JSONL label file where each line is a record like
    ``{"tools": [...], "messages": [{"role": "user", "content": "..."}], "label": "<tool>"}``.
    Each record becomes a QAPair whose ``query`` is the user message and whose
    single answer names the labelled tool (with empty arguments, since the raw
    label data carries no parameters). These serve as authentic few-shot
    references when generating new queries for the tool.
    """

    def __init__(
        self,
        examples_by_tool: dict[str, list[QAPair]],
        count_range: tuple[int, int],
        rng: random.Random | None = None,
    ) -> None:
        self._by_tool = examples_by_tool
        self._range = count_range
        self._rng = rng or random.Random()

    def has_examples(self, tool: str) -> bool:
        return bool(self._by_tool.get(tool))

    def tools_with_examples(self) -> list[str]:
        return [t for t, ex in self._by_tool.items() if ex]

    def sample_tool(self) -> str | None:
        """Return a random tool name that has at least one real example."""
        tools = self.tools_with_examples()
        return self._rng.choice(tools) if tools else None

    def sample_for(self, tool_names: list[str]) -> list[QAPair]:
        """Sample real examples whose tool is among ``tool_names``.

        Returns an empty list when none of the requested tools have examples.
        """
        pool: list[QAPair] = []
        for name in tool_names:
            pool.extend(self._by_tool.get(name, ()))
        if not pool:
            return []
        lo, hi = self._range
        hi = min(hi, len(pool))
        lo = min(lo, hi)
        k = self._rng.randint(lo, hi) if hi >= lo else 0
        return self._rng.sample(pool, k)

    # ── Loaders ──────────────────────────────────────────────────────────────

    @staticmethod
    def _last_user_message(record: dict) -> str | None:
        messages = record.get("messages") or []
        for msg in reversed(messages):
            if msg.get("role") == "user" and msg.get("content", "").strip():
                return msg["content"].strip()
        return None

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        count_range: tuple[int, int] = (1, 3),
        valid_tools: set[str] | None = None,
        rng: random.Random | None = None,
    ) -> "LabelExampleSampler":
        """Load real labelled examples from a JSONL file.

        ``valid_tools`` (e.g. the API library's names) filters out records whose
        label is not an available tool, so few-shot examples never reference a
        function the pipeline cannot generate.
        """
        by_tool: dict[str, list[QAPair]] = {}
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                label = record.get("label")
                query = cls._last_user_message(record)
                if not label or not query:
                    continue
                if valid_tools is not None and label not in valid_tools:
                    continue
                pair = QAPair(query=query, answers=[FunctionCall(name=label)])
                by_tool.setdefault(label, []).append(pair)
        return cls(by_tool, count_range, rng=rng)
