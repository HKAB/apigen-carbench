"""Execution backend for local Python car functions."""

from __future__ import annotations

from typing import Any, Callable

from ..car_functions import CAR_FUNCTIONS, VehicleState
from ..schemas import FunctionCall
from .base import ExecutionBackend, ExecutionError


class PythonExecutionBackend(ExecutionBackend):
    """Dispatches calls to in-process Python functions over a VehicleState.

    A fresh VehicleState is used per call by default so executions are
    independent (the paper executes each generated answer in isolation).
    """

    def __init__(self, functions: dict[str, Callable] | None = None):
        self._functions = functions if functions is not None else dict(CAR_FUNCTIONS)

    def supports(self, name: str) -> bool:
        return name in self._functions

    def execute(self, call: FunctionCall) -> Any:
        func = self._functions.get(call.name)
        if func is None:
            raise ExecutionError(f"no executable backend for function '{call.name}'")
        state = VehicleState()
        try:
            return func(state, **call.arguments)
        except TypeError as exc:
            # Wrong/missing/extra arguments surface as TypeError from the call.
            raise ExecutionError(f"invalid arguments for '{call.name}': {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - capture fine-grained message
            raise ExecutionError(f"execution of '{call.name}' failed: {exc}") from exc
