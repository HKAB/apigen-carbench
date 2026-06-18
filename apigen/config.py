"""Centralized configuration, loaded from environment / .env.

All LLM usage is decoupled and routed to an external vLLM server; the four
README-mandated variables live here, plus a few pipeline knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


def _float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


@dataclass
class Settings:
    # --- vLLM serving layer (the four README variables) ---
    llm_base_url: str = "http://localhost:8000/v1"
    llm_api_token: str = "EMPTY"
    generator_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    semantic_checker_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # --- pipeline knobs ---
    generation_temperature: float = 0.7
    concurrency: int = 8
    num_apis_range: tuple[int, int] = (1, 4)
    num_seed_range: tuple[int, int] = (1, 3)

    # --- data locations ---
    apis_path: str = "data/apis/car_apis.json"
    seed_path: str = "data/seed/seed_qa.json"
    # Optional real-label few-shot source (JSONL); None disables the feature.
    label_path: str | None = None

    @classmethod
    def from_env(cls, *, load: bool = True) -> "Settings":
        if load:
            load_dotenv()
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_api_token=os.getenv("LLM_API_TOKEN", cls.llm_api_token),
            generator_model_name=os.getenv("GENERATOR_MODEL_NAME", cls.generator_model_name),
            semantic_checker_model_name=os.getenv(
                "SEMANTIC_CHECKER_MODEL_NAME", cls.semantic_checker_model_name
            ),
            generation_temperature=_float("APIGEN_GENERATION_TEMPERATURE", cls.generation_temperature),
            concurrency=_int("APIGEN_CONCURRENCY", cls.concurrency),
            num_apis_range=(
                _int("APIGEN_NUM_APIS_MIN", 1),
                _int("APIGEN_NUM_APIS_MAX", 4),
            ),
            num_seed_range=(
                _int("APIGEN_NUM_SEED_MIN", 1),
                _int("APIGEN_NUM_SEED_MAX", 3),
            ),
            label_path=os.getenv("APIGEN_LABEL_FILE") or None,
        )
