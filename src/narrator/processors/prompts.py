"""Explainer prompts (ARCHITECTURE.md sections 6 / 15-D).

Prompts live ONLY here. MAP emits strict JSON; REDUCE emits plain spoken text
with ``[pause:ms]`` markers only (no markdown / bullets / citations), verbalized
numbers, hook -> body -> recap. A stricter regeneration prompt is used once when
the grounding validator finds unverified terms.
"""

from __future__ import annotations

import json
from typing import Any

MAP_INSTRUCTIONS = (
    "You extract the substance of one section of a document. "
    "Respond with STRICT JSON only, no prose, no markdown fences. Schema:\n"
    '{"key_points": [string, ...], "entities": [string, ...], "numbers": [string, ...]}\n'
    "- key_points: the load-bearing facts/claims of THIS section, in plain words.\n"
    "- entities: proper nouns (people, orgs, places, products) that appear.\n"
    "- numbers: every figure/stat/date exactly as written."
)

REDUCE_STYLE = {
    "news": (
        "Write like a clear, neutral news explainer for a general audience. "
        "Open with a one-line hook, explain the body plainly, end with a short recap."
    ),
    "teacher": (
        "Write like a patient teacher. Define terms simply, build understanding "
        "step by step, use short analogies, end by restating the key takeaways."
    ),
    "podcast": (
        "Write like a warm, conversational solo podcast host. Natural spoken "
        "rhythm, light signposting, a friendly hook and a wrap-up."
    ),
}

_REDUCE_RULES = (
    "Output rules (STRICT):\n"
    "- Plain spoken text ONLY. No markdown, no bullet points, no headings, no citations, no URLs.\n"
    "- Verbalize ALL numbers and symbols (say 'twenty-five percent', not '25%').\n"
    "- Insert pauses ONLY as literal markers like [pause:600] between beats (200-900 ms typical).\n"
    "- Do not invent facts, names, or numbers: use only what the key points provide.\n"
    "- Structure: hook, then body, then a short recap."
)


def build_map_prompt(section_text: str) -> str:
    return f"{MAP_INSTRUCTIONS}\n\n--- SECTION START ---\n{section_text}\n--- SECTION END ---"


def _facts_blob(maps: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "key_points": [kp for m in maps for kp in m.get("key_points", [])],
            "entities": sorted({e for m in maps for e in m.get("entities", [])}),
            "numbers": sorted({str(n) for m in maps for n in m.get("numbers", [])}),
        },
        ensure_ascii=False,
    )


def build_reduce_prompt(style: str, maps: list[dict[str, Any]], target_minutes: int) -> str:
    persona = REDUCE_STYLE.get(style, REDUCE_STYLE["news"])
    words = int(target_minutes * 155)  # ~155 wpm narration
    return (
        f"{persona}\n\n{_REDUCE_RULES}\n\n"
        f"Aim for about {words} words (~{target_minutes} minutes of speech).\n\n"
        f"Here are the verified facts to narrate (JSON):\n{_facts_blob(maps)}\n\n"
        "Write the narration now:"
    )


def build_regen_prompt(
    style: str,
    maps: list[dict[str, Any]],
    target_minutes: int,
    unverified: list[str],
) -> str:
    base = build_reduce_prompt(style, maps, target_minutes)
    flagged = ", ".join(sorted(unverified)[:20])
    return (
        f"{base}\n\nIMPORTANT: A previous draft introduced terms not present in "
        f"the source facts: {flagged}. Rewrite using ONLY the facts above; remove "
        "or generalize anything not directly supported."
    )
