"""Runs the three verification stages in order, short-circuiting on failure."""

from __future__ import annotations

from ..api_library import ApiLibrary
from ..schemas import APIDef, GeneratorOutput, VerifyResult
from .execution_checker import ExecutionChecker
from .format_checker import FormatChecker
from .semantic_checker import SemanticChecker


class VerificationPipeline:
    def __init__(
        self,
        format_checker: FormatChecker,
        execution_checker: ExecutionChecker,
        semantic_checker: SemanticChecker,
        library: ApiLibrary,
    ): 
        self._format = format_checker
        self._execution = execution_checker
        self._semantic = semantic_checker
        self._library = library

    def _apis_for(self, item: GeneratorOutput) -> list[APIDef]:
        seen: dict[str, APIDef] = {}
        for call in item.answers:
            api = self._library.get(call.name)
            if api is not None:
                seen[api.name] = api
        return list(seen.values())

    async def verify(self, item: GeneratorOutput) -> VerifyResult:
        # Stage 1: format (fast, LLM-free).
        result = self._format.check(item)
        if not result.passed:
            return result

        # Stage 2: execution.
        result = await self._execution.check(item)
        if not result.passed:
            return result
        execution_results = result.execution_results

        # Stage 3: semantic (LLM).
        return await self._semantic.check(item, self._apis_for(item), execution_results)
