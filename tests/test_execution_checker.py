import pytest

from apigen.backends import PythonExecutionBackend
from apigen.schemas import FunctionCall, GeneratorOutput
from apigen.verification import ExecutionChecker


def _item(name, arguments):
    return GeneratorOutput(query="q", answers=[FunctionCall(name=name, arguments=arguments)])


@pytest.fixture
def checker():
    return ExecutionChecker(PythonExecutionBackend())


async def test_valid_call_executes(checker):
    res = await checker.check(_item("set_volume", {"level": 20}))
    assert res.passed
    assert res.execution_results[0]["result"]["volume"] == 20


async def test_runtime_error_fails(checker):
    # volume out of range raises ValueError in the mock function.
    res = await checker.check(_item("set_volume", {"level": 999}))
    assert not res.passed and res.failed_stage == "execution"
    assert "volume" in res.reason


async def test_unsupported_function_fails(checker):
    res = await checker.check(_item("unknown_fn", {}))
    assert not res.passed and res.failed_stage == "execution"


async def test_multiple_calls_all_execute(checker):
    item = GeneratorOutput(
        query="q",
        answers=[
            FunctionCall(name="set_volume", arguments={"level": 10}),
            FunctionCall(name="lock_doors", arguments={"lock": True}),
        ],
    )
    res = await checker.check(item)
    assert res.passed and len(res.execution_results) == 2
