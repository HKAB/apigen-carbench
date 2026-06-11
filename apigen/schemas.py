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
