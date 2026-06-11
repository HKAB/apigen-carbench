"""Standardized JSON schemas shared across the pipeline.

These mirror the three formats from the APIGen paper figures:
  - API def       : {name, description, parameters: {param: {type, description, default, required}}}
  - Function call : {name, arguments: {arg: value}}
  - Generator out : {query, answers: [function call, ...]}
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Query styles from the paper (Query Style Diversity).
QueryStyle = Literal["simple", "multiple", "parallel", "parallel_multiple"]

# Which verification stage a data point reached / failed at.
Stage = Literal["format", "execution", "semantic"]


class Parameter(BaseModel):
    """A single API parameter description."""

    type: str
    description: str = ""
    default: Any = None
    required: bool = False


class APIDef(BaseModel):
    """Standardized API definition (the 'APIs' format in the paper)."""

    name: str
    description: str = ""
    parameters: dict[str, Parameter] = Field(default_factory=dict)

    def required_params(self) -> list[str]:
        return [name for name, p in self.parameters.items() if p.required]


class FunctionCall(BaseModel):
    """A single function call (the 'Function Call' format in the paper)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class GeneratorOutput(BaseModel):
    """One generated query-answer pair (the 'Generator Output' format)."""

    model_config = ConfigDict(extra="forbid")

    query: str
    answers: list[FunctionCall] = Field(default_factory=list)


class QAPair(BaseModel):
    """A query-answer pair as stored in seed data / output.

    Identical shape to GeneratorOutput but kept distinct so seed data and
    verified output can carry optional metadata without polluting the
    strict generator-output contract.
    """

    query: str
    answers: list[FunctionCall] = Field(default_factory=list)
    style: QueryStyle | None = None

    def to_generator_output(self) -> GeneratorOutput:
        return GeneratorOutput(query=self.query, answers=self.answers)


class Persona(BaseModel):
    """One row from nvidia/Nemotron-Personas-Vietnam (21 fields).

    All narrative/string fields are optional so the model tolerates
    partially-filled rows that may appear in the dataset.
    """

    model_config = ConfigDict(extra="ignore")  # ignore unknown keys from HF rows

    # ── Identifiers ────────────────────────────────────────────────────────
    uuid: str = ""

    # ── 6 Persona narratives ───────────────────────────────────────────────
    professional_persona: str | None = None
    sports_persona: str | None = None
    arts_persona: str | None = None
    travel_persona: str | None = None
    culinary_persona: str | None = None
    persona: str | None = None

    # ── 15 Contextual / demographic fields ────────────────────────────────
    cultural_background: str | None = None
    skills_and_expertise: str | None = None
    skills_and_expertise_list: str | None = None
    hobbies_and_interests: str | None = None
    hobbies_and_interests_list: str | None = None
    career_goals_and_ambitions: str | None = None
    sex: str | None = None                 # Nam / Nữ
    age: int | None = None                 # 18–90
    marital_status: str | None = None      # Độc thân / Đã kết hôn / Góa / Ly thân
    education_level: str | None = None
    occupation: str | None = None
    zone: str | None = None                # Đô Thị / Nông Thôn
    region: str | None = None              # one of 6 provinces/cities
    country: str | None = None             # Việt Nam (constant)


class VerifyResult(BaseModel):
    """Outcome of running a data point through the verification pipeline."""

    passed: bool
    # The stage that caused a failure (None when passed is True).
    failed_stage: Stage | None = None
    reason: str = ""
    # Execution results captured by the execution checker (one per answer).
    execution_results: list[Any] = Field(default_factory=list)

    @classmethod
    def ok(cls, execution_results: list[Any]) -> "VerifyResult":
        return cls(passed=True, execution_results=execution_results)

    @classmethod
    def fail(cls, stage: Stage, reason: str) -> "VerifyResult":
        return cls(passed=False, failed_stage=stage, reason=reason)
