"""Stage 1: Format Checker.

Fast, LLM-free sanity checks. Validates the generated structure and ensures
every function call references a real API with valid, complete arguments.
Rejects hallucinated functions/arguments and missing required parameters.
"""

from __future__ import annotations

from typing import Any

from ..api_library import ApiLibrary
from ..schemas import APIDef, GeneratorOutput, VerifyResult

# Loose JSON-type compatibility map: API type name -> acceptable Python types.
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "str": (str,),
    "integer": (int,),
    "int": (int,),
    # JSON has no int/float distinction; accept int for number too.
    "number": (int, float),
    "float": (int, float),
    "boolean": (bool,),
    "bool": (bool,),
    "array": (list,),
    "list": (list,),
    "object": (dict,),
    "dict": (dict,),
}


def _type_ok(expected: str, value: Any) -> bool:
    allowed = _TYPE_MAP.get(expected.lower())
    if allowed is None:
        return True  # unknown declared type -> don't block
    # bool is a subclass of int; guard so a bool isn't accepted as integer/number.
    if bool not in allowed and isinstance(value, bool):
        return False
    return isinstance(value, allowed)


class FormatChecker:
    def __init__(self, library: ApiLibrary):
        self._library = library

    def check(self, item: GeneratorOutput) -> VerifyResult:
        if not item.query or not item.query.strip():
            return VerifyResult.fail("format", "empty query")
        if not item.answers:
            return VerifyResult.fail("format", "no answers provided")

        for call in item.answers:
            api: APIDef | None = self._library.get(call.name)
            if api is None:
                return VerifyResult.fail("format", f"hallucinated function '{call.name}'")

            declared = set(api.parameters)
            provided = set(call.arguments)

            unknown = provided - declared
            if unknown:
                return VerifyResult.fail(
                    "format", f"unknown argument(s) {sorted(unknown)} for '{call.name}'"
                )

            missing = [p for p in api.required_params() if p not in provided]
            if missing:
                return VerifyResult.fail(
                    "format", f"missing required parameter(s) {missing} for '{call.name}'"
                )

            for arg_name, value in call.arguments.items():
                expected = api.parameters[arg_name].type
                if not _type_ok(expected, value):
                    return VerifyResult.fail(
                        "format",
                        f"argument '{arg_name}' of '{call.name}' expected {expected}, got {type(value).__name__}",
                    )

        return VerifyResult.ok(execution_results=[])
