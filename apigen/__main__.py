"""CLI entry point: wire everything together and run the pipeline live."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from .api_library import ApiLibrary
from .backends import PythonExecutionBackend
from .config import Settings
from .feedback import FeedbackLoop
from .generator import QueryAnswerGenerator
from .llm_client import AiohttpLLMClient
from .orchestrator import Orchestrator
from .samplers import ApiSampler, PromptSampler, SeedQASampler, PersonaSampler
from .schemas import QAPair
from .verification import (
    ExecutionChecker,
    FormatChecker,
    SemanticChecker,
    VerificationPipeline,
)


def _load_seeds(path: str) -> list[QAPair]:
    raw = json.loads(Path(path).read_text())
    return [QAPair.model_validate(item) for item in raw]


def _build(
    settings: Settings,
    client,
    seed: int | None,
    semantic_fail_log_path: str | None = None,
    verified_log_path: str | None = None,
):
    rng = random.Random(seed)
    library = ApiLibrary.from_json(settings.apis_path)
    seeds = _load_seeds(settings.seed_path)

    api_sampler = ApiSampler(library.all(), settings.num_apis_range, rng)
    seed_sampler = SeedQASampler(seeds, settings.num_seed_range, rng)
    prompt_sampler = PromptSampler(rng=rng)

    generator = QueryAnswerGenerator(
        client, settings.generator_model_name, settings.generation_temperature
    )
    persona_sampler = PersonaSampler.from_huggingface()
    backend = PythonExecutionBackend()
    pipeline = VerificationPipeline(
        FormatChecker(library),
        ExecutionChecker(backend),
        SemanticChecker(client, settings.semantic_checker_model_name),
        library,
    )
    feedback = FeedbackLoop(seed_sampler)
    return Orchestrator(
        generator,
        pipeline,
        api_sampler,
        seed_sampler,
        prompt_sampler,
        persona_sampler,
        feedback,
        concurrency=settings.concurrency,
        semantic_fail_log_path=semantic_fail_log_path,
        verified_log_path=verified_log_path,
    )


async def _run(args) -> None:
    settings = Settings.from_env()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Start a fresh output file, then append verified pairs continuously during the run.
    out_path.write_text("", encoding="utf-8")

    async with AiohttpLLMClient(settings.llm_base_url, settings.llm_api_token) as client:
        orch = _build(
            settings,
            client,
            args.seed,
            args.semantic_fail_log,
            str(out_path),
        )
        if args.concurrency:
            orch._concurrency = args.concurrency
        verified = await orch.run(args.num)

    print(f"[apigen] {orch.stats.summary()}")
    print(f"[apigen] wrote {len(verified)} verified pairs to {out_path}")
    if args.semantic_fail_log:
        print(f"[apigen] wrote semantic rejects to {args.semantic_fail_log}")


def main() -> None:
    parser = argparse.ArgumentParser(description="APIGen pipeline (user-car domain)")
    parser.add_argument("--num", type=int, default=20, help="target verified pairs")
    parser.add_argument("--concurrency", type=int, default=0, help="override concurrency")
    parser.add_argument("--out", default="data/output/verified.jsonl", help="output JSONL path")
    parser.add_argument(
        "--semantic-fail-log",
        default="data/output/semantic_rejects.jsonl",
        help="JSONL path for stage-3 semantic rejects",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
