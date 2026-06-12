"""Prompt templates for the four query styles.

Each builds a chat prompt that steers the generator LLM to emit a JSON array of
{query, answers} objects (the batching trick: many pairs per inference).
"""

from __future__ import annotations

import json

from .schemas import APIDef, Persona, QAPair, QueryStyle

STYLE_INSTRUCTIONS: dict[QueryStyle, str] = {
    "simple": (
        "Generate queries that each require exactly ONE function call to satisfy. "
        "Use a single API per query."
    ),
    "multiple": (
        "You are given several APIs. Generate queries where each query is satisfied by exactly ONE "
        "function call, but the correct API must be chosen from among the several provided."
    ),
    "parallel": (
        "Generate queries that each require MULTIPLE function calls (to the same or different APIs) "
        "executed together to be fully satisfied. Each query's 'answers' list must contain more than one call."
    ),
    "parallel_multiple": (
        "You are given several APIs. Generate queries that each require MULTIPLE function calls drawn "
        "from the different provided APIs, all needed together to satisfy the request."
    ),
}

SYSTEM_PROMPT = (
    "You are a data generator for a user-car voice-assistant function-calling dataset. "
    "Given a set of car-control APIs, you produce realistic natural-language Vietnamese driver queries paired "
    "with the exact function calls that fulfill them. You only use the provided APIs and their "
    "declared parameters. You never invent functions or arguments."
)

OUTPUT_CONTRACT = """\
Respond with ONLY a JSON array (no prose, no markdown fences). Each element must be an object:
  {"query": "<natural language request>", "answers": [{"name": "<api_name>", "arguments": {"<arg>": <value>}}]}
Rules:
- Use only the API names and parameter names listed above.
- Include every required parameter; use values within any stated valid range.
- "arguments" values must be concrete (numbers, strings, booleans), never placeholders.
"""


def _format_apis(apis: list[APIDef]) -> str:
    return json.dumps([api.model_dump() for api in apis], indent=2)


def _format_seeds(seeds: list[QAPair]) -> str:
    if not seeds:
        return "(none)"
    return json.dumps(
        [{"query": s.query, "answers": [a.model_dump() for a in s.answers]} for s in seeds],
        indent=2,
    )


def _format_persona(p: Persona) -> str:
    """Render a Persona into a compact, LLM-readable profile block.

    Fields are omitted when empty so the block stays concise for personas
    that only have partial data.
    """
    lines: list[str] = []

    # ─ Demographics row ────────────────────────────────────────────────────────────────
    demo: list[str] = []
    if p.sex:
        demo.append(p.sex)
    if p.age is not None:
        demo.append(f"{p.age} tuổi")
    if p.occupation:
        demo.append(p.occupation)
    if p.marital_status:
        demo.append(p.marital_status)
    if p.education_level:
        demo.append(p.education_level)
    if demo:
        lines.append("▸ Hồ sơ: " + " | ".join(demo))

    # ─ Location ───────────────────────────────────────────────────────────────────
    loc: list[str] = [x for x in (p.zone, p.region, p.country) if x]
    if loc:
        lines.append("▸ Khu vực: " + ", ".join(loc))

    # ─ General persona narrative ─────────────────────────────────────────────────────
    if p.persona:
        lines.append(f"▸ Tóm tắt: {p.persona}")

    # ─ Specialised persona narratives ──────────────────────────────────────────────
    for label, value in [
        ("Công việc", p.professional_persona),
        ("Ẩm thực", p.culinary_persona),
        ("Du lịch", p.travel_persona),
        ("Thể thao", p.sports_persona),
        ("Nghệ thuật", p.arts_persona),
    ]:
        if value:
            lines.append(f"▸ {label}: {value}")

    # ─ Extra contextual details ─────────────────────────────────────────────────────
    if p.hobbies_and_interests:
        lines.append(f"▸ Sở thích: {p.hobbies_and_interests}")
    if p.skills_and_expertise:
        lines.append(f"▸ Kỹ năng: {p.skills_and_expertise}")
    if p.cultural_background:
        lines.append(f"▸ Văn hóa: {p.cultural_background}")
    if p.career_goals_and_ambitions:
        lines.append(f"▸ Mục tiêu: {p.career_goals_and_ambitions}")

    return "\n".join(lines)


def build_messages(
    style: QueryStyle,
    apis: list[APIDef],
    seeds: list[QAPair],
    num_pairs: int,
    persona: Persona | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages for one generation request.

    When a ``persona`` is supplied the user turn includes a ``## User Persona``
    section that instructs the LLM to imitate that specific Vietnamese driver,
    producing queries that are authentic to their demographics, vocabulary, and
    cultural context.
    """
    persona_block = ""
    if persona is not None:
        persona_block = f"""\n## User Persona
Generate queries AS IF spoken by this specific Vietnamese car owner.
Imitate their vocabulary, tone, cultural references, and daily-life situations.
The queries must feel natural and authentic to who this person is —
not generic textbook examples.

Remember that the user are talking with a voice assistant, 
so they might call the car by its name (e.g, "em", "VinFast", "bạn", "cháu", etc.) —
but not any specific human names, since the car is not a human. Sometimes they just gave direct order.

{_format_persona(persona)}
"""

    user = f"""\
## Available APIs
{_format_apis(apis)}

## Example query-answer pairs (for style reference)
{_format_seeds(seeds)}
{persona_block}
## Task
{STYLE_INSTRUCTIONS[style]}
Generate {num_pairs} diverse query-answer pair(s) in this style.

## Output format
{OUTPUT_CONTRACT}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
