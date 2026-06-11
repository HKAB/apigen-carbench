"""Execution backend plugin seam.

Stage 2 (execution checker) runs function calls against a backend. New API
types (REST, GraphQL, ...) are added by implementing this ABC, without
touching the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..schemas import FunctionCall


class ExecutionError(Exception):
    """Raised when a function call cannot be executed successfully."""


class ExecutionBackend(ABC):
    """Executes a single function call, returning its result or raising."""

    @abstractmethod
    def supports(self, name: str) -> bool:
        """Whether this backend can execute the named function."""

    @abstractmethod
    def execute(self, call: FunctionCall) -> Any:
        """Execute the call and return a JSON-serializable result.

        Must raise ExecutionError (or any Exception) on failure so the
        execution checker can capture a fine-grained error message.
        """
