"""Stage 2: Execution Checker.

Runs each well-formatted function call against an execution backend. Any
failure (unsupported function, bad arguments, runtime error) filters the data
point out with a fine-grained error message.
"""

from __future__ import annotations

import asyncio

from ..backends.base import ExecutionBackend
from ..schemas import GeneratorOutput, VerifyResult


class ExecutionChecker:
    def __init__(self, backend: ExecutionBackend):
        self._backend = backend

    async def check(self, item: GeneratorOutput) -> VerifyResult:
        results = []
        for call in item.answers:
            if not self._backend.supports(call.name):
                return VerifyResult.fail("execution", f"no backend supports '{call.name}'")
            try:
                # Backends are synchronous; offload so we never block the loop.
                result = await asyncio.to_thread(self._backend.execute, call)
            except Exception as exc:  # noqa: BLE001 - capture fine-grained message
                return VerifyResult.fail("execution", f"{call.name}: {exc}")
            results.append({"name": call.name, "result": result})
        return VerifyResult.ok(execution_results=results)
