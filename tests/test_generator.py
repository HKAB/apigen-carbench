from apigen.generator import QueryAnswerGenerator, parse_generator_output
from tests.conftest import FakeLLMClient

MODEL = "gen-model"


def test_parse_plain_array():
    text = '[{"query": "lock doors", "answers": [{"name": "lock_doors", "arguments": {"lock": true}}]}]'
    out = parse_generator_output(text)
    assert len(out) == 1
    assert out[0].answers[0].name == "lock_doors"


def test_parse_fenced_json():
    text = '```json\n[{"query": "q", "answers": [{"name": "set_volume", "arguments": {"level": 5}}]}]\n```'
    out = parse_generator_output(text)
    assert len(out) == 1 and out[0].answers[0].arguments["level"] == 5


def test_parse_with_surrounding_prose():
    text = 'Sure! Here you go:\n[{"query": "q", "answers": []}]\nHope that helps.'
    out = parse_generator_output(text)
    assert len(out) == 1


def test_parse_skips_malformed_items():
    text = '[{"query": "ok", "answers": []}, {"not_a_field": 1}]'
    out = parse_generator_output(text)
    assert len(out) == 1


def test_parse_garbage_returns_empty():
    assert parse_generator_output("no json here") == []


async def test_generator_calls_client(library):
    resp = '[{"query": "set fan to 3", "answers": [{"name": "set_fan_speed", "arguments": {"level": 3}}]}]'
    client = FakeLLMClient({MODEL: [resp]})
    gen = QueryAnswerGenerator(client, MODEL, temperature=0.7)
    out = await gen.generate(library.all(), [], "simple", num_pairs=1)
    assert len(out) == 1
    assert client.calls[0]["model"] == MODEL
    assert client.calls[0]["temperature"] == 0.7
