from apigen.schemas import FunctionCall, GeneratorOutput
from apigen.verification import FormatChecker


def _item(name, arguments, query="do it"):
    return GeneratorOutput(query=query, answers=[FunctionCall(name=name, arguments=arguments)])


def test_valid_call_passes(library):
    checker = FormatChecker(library)
    item = _item("set_climate_temperature", {"zone": "driver", "temperature": 22})
    assert checker.check(item).passed


def test_hallucinated_function_fails(library):
    checker = FormatChecker(library)
    res = checker.check(_item("teleport_car", {"x": 1}))
    assert not res.passed and res.failed_stage == "format"
    assert "hallucinated" in res.reason


def test_unknown_argument_fails(library):
    checker = FormatChecker(library)
    res = checker.check(_item("set_fan_speed", {"level": 3, "bogus": 1}))
    assert not res.passed and res.failed_stage == "format"
    assert "unknown argument" in res.reason


def test_missing_required_parameter_fails(library):
    checker = FormatChecker(library)
    # set_climate_temperature requires both zone and temperature.
    res = checker.check(_item("set_climate_temperature", {"zone": "driver"}))
    assert not res.passed and res.failed_stage == "format"
    assert "missing required" in res.reason


def test_wrong_type_fails(library):
    checker = FormatChecker(library)
    res = checker.check(_item("set_fan_speed", {"level": "high"}))
    assert not res.passed and res.failed_stage == "format"
    assert "expected" in res.reason


def test_bool_not_accepted_as_integer(library):
    checker = FormatChecker(library)
    res = checker.check(_item("set_fan_speed", {"level": True}))
    assert not res.passed and res.failed_stage == "format"


def test_empty_answers_fails(library):
    checker = FormatChecker(library)
    res = checker.check(GeneratorOutput(query="hi", answers=[]))
    assert not res.passed and res.failed_stage == "format"
