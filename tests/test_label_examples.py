"""Tests for the real-label few-shot feature."""

import json
import random

from apigen.prompts import build_messages
from apigen.samplers import ApiSampler, LabelExampleSampler
from apigen.schemas import APIDef, QAPair

LABEL_LINES = [
    {"id": "a", "messages": [{"role": "user", "content": "Kim Ngưu không hợp với cung nào"}], "label": "zodiac_search"},
    {"id": "b", "messages": [{"role": "user", "content": "Con số may mắn của Song Ngư"}], "label": "zodiac_search"},
    {"id": "c", "messages": [{"role": "user", "content": "Trời mai có mưa không"}], "label": "weather_tool"},
    {"id": "d", "messages": [{"role": "system", "content": "x"}], "label": "weather_tool"},  # no user msg -> skipped
    {"id": "e", "messages": [{"role": "user", "content": "hi"}], "label": "not_a_real_tool"},  # filtered by valid_tools
]


def _write_jsonl(tmp_path):
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in LABEL_LINES), encoding="utf-8")
    return p


def test_from_jsonl_indexes_by_tool_and_filters(tmp_path):
    path = _write_jsonl(tmp_path)
    sampler = LabelExampleSampler.from_jsonl(
        path, count_range=(1, 2), valid_tools={"zodiac_search", "weather_tool"}
    )
    # zodiac_search has 2; weather_tool has 1 (the system-only record is skipped).
    assert len(sampler._by_tool["zodiac_search"]) == 2
    assert len(sampler._by_tool["weather_tool"]) == 1
    # not_a_real_tool was filtered out by valid_tools.
    assert "not_a_real_tool" not in sampler._by_tool


def test_query_is_user_message_with_label_as_answer(tmp_path):
    path = _write_jsonl(tmp_path)
    sampler = LabelExampleSampler.from_jsonl(path, valid_tools={"zodiac_search", "weather_tool"})
    ex = sampler._by_tool["zodiac_search"][0]
    assert ex.query == "Kim Ngưu không hợp với cung nào"
    assert ex.answers[0].name == "zodiac_search"
    assert ex.answers[0].arguments == {}


def test_sample_for_filters_by_tool(tmp_path):
    path = _write_jsonl(tmp_path)
    sampler = LabelExampleSampler.from_jsonl(
        path, count_range=(2, 2), valid_tools={"zodiac_search", "weather_tool"},
        rng=random.Random(0),
    )
    got = sampler.sample_for(["zodiac_search"])
    assert got and all(e.answers[0].name == "zodiac_search" for e in got)
    # A tool with no examples yields nothing.
    assert sampler.sample_for(["set_volume"]) == []


def test_sample_tool_returns_known_tool(tmp_path):
    path = _write_jsonl(tmp_path)
    sampler = LabelExampleSampler.from_jsonl(
        path, valid_tools={"zodiac_search", "weather_tool"}, rng=random.Random(1)
    )
    assert sampler.sample_tool() in {"zodiac_search", "weather_tool"}


def test_api_sampler_force_includes():
    apis = [APIDef(name=f"f{i}") for i in range(5)]
    sampler = ApiSampler(apis, (1, 1), random.Random(0))
    chosen = sampler.sample(include=["f4"])
    assert any(a.name == "f4" for a in chosen)


def test_build_messages_renders_real_examples_section():
    apis = [APIDef(name="zodiac_search", description="zodiac")]
    real = [QAPair(query="Kim Ngưu hợp cung nào", answers=[{"name": "zodiac_search"}])]
    msgs = build_messages("simple", apis, seeds=[], num_pairs=1, real_examples=real)
    content = msgs[-1]["content"]
    assert "Real user query examples" in content
    assert "Kim Ngưu hợp cung nào" in content
    assert "→ tool: zodiac_search" in content
    # Reminds the model to still emit full arguments.
    assert "COMPLETE function call" in content


def test_build_messages_omits_section_when_no_real_examples():
    apis = [APIDef(name="zodiac_search")]
    msgs = build_messages("simple", apis, seeds=[], num_pairs=1)
    assert "Real user query examples" not in msgs[-1]["content"]
