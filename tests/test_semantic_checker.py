from apigen.schemas import FunctionCall, GeneratorOutput
from apigen.verification import SemanticChecker
from tests.conftest import FakeLLMClient

MODEL = "checker-model"


def _item():
    return GeneratorOutput(
        query="set volume to 10",
        answers=[FunctionCall(name="set_volume", arguments={"level": 10})],
    )


async def test_semantic_pass(library):
    client = FakeLLMClient({MODEL: ['{"pass": true, "reason": "matches"}']})
    checker = SemanticChecker(client, MODEL)
    res = await checker.check(_item(), library.all(), [{"name": "set_volume", "result": {"volume": 10}}])
    assert res.passed


async def test_semantic_fail(library):
    client = FakeLLMClient({MODEL: ['{"pass": false, "reason": "volume wrong"}']})
    checker = SemanticChecker(client, MODEL)
    res = await checker.check(_item(), library.all(), [])
    assert not res.passed and res.failed_stage == "semantic"
    assert "volume wrong" in res.reason


async def test_semantic_handles_fenced_json(library):
    client = FakeLLMClient({MODEL: ['```json\n{"pass": true, "reason": "ok"}\n```']})
    checker = SemanticChecker(client, MODEL)
    res = await checker.check(_item(), library.all(), [])
    assert res.passed


async def test_semantic_unparseable_fails(library):
    client = FakeLLMClient({MODEL: ["I think it's fine honestly"]})
    checker = SemanticChecker(client, MODEL)
    res = await checker.check(_item(), library.all(), [])
    assert not res.passed and res.failed_stage == "semantic"
