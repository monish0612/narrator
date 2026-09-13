"""Explainer prompts. Structural constraints, not exact word-count trust.

AI News uses a two-call relevance-check-then-generate pattern comparing to
RPA / UiPath only when a genuine parallel exists. Every other category uses
the plain layman's explainer with no comparison instruction.
"""

from __future__ import annotations

AI_NEWS_CATEGORY = "AI News"

_PLAIN_RULES = (
    "Output rules (STRICT):\n"
    "- Plain spoken English ONLY. No markdown, bullets, headings, citations, or URLs.\n"
    "- Verbalize numbers and symbols ('twenty-five percent', not '25%').\n"
    "- Do not invent facts, names, or numbers that are not in the article.\n"
    "- Structure: a one-sentence hook, then the body, then a short recap, "
    "then a closer that makes it obvious you are finished.\n"
    "- Write 8 to 12 short paragraphs. Each paragraph is 2 to 4 sentences.\n"
    "- Aim for spoken length of about nine minutes at a calm pace "
    "(roughly thirteen hundred to fourteen hundred and fifty words). "
    "Prefer slightly short over rambling.\n"
    "- Write the listener-facing script directly. No hidden reasoning, "
    "planning notes, or think tags.\n"
    "- Always end with one or two short spoken sentences so the listener "
    "knows the article ended, not that playback got stuck. No new facts "
    "in that closer. Tone: 'That's it from this article. I'll leave it there.'"
)

PLAIN_EXPLAINER = (
    "You write a spoken news explainer for a general listener. "
    "Assume they are smart but not an expert in this field. "
    "Define jargon in passing the first time it appears.\n\n"
    f"{_PLAIN_RULES}"
)

AI_RELEVANCE_SYSTEM = (
    "You decide whether an AI-news article has a genuine, non-forced parallel "
    "to robotic process automation (RPA) or UiPath. Respond with STRICT JSON only, "
    "no markdown fences. Schema:\n"
    '{"relevant": true|false, "reason": "one sentence", "parallel": "short or empty"}\n'
    "relevant=true ONLY when the article's actual subject maps onto automation, "
    "agents that do work, document understanding, orchestration, attended/unattended "
    "bots, process mining, or enterprise workflow — not merely because both involve software.\n\n"
    "Few-shot examples:\n"
    "1) Article: UiPath Autopilot and agentic automation in the enterprise. "
    '{"relevant": true, "reason": "It is literally enterprise automation software.", '
    '"parallel": "Same job as an unattended robot plus a planner."}\n'
    "2) Article: A new OpenAI image model with no workflow or operations angle. "
    '{"relevant": false, "reason": "A research/model release with no operations parallel.", '
    '"parallel": ""}\n'
    "3) Article: Document understanding / IDP extracting invoices at scale. "
    '{"relevant": true, "reason": "IDP is a core RPA document-processing workload.", '
    '"parallel": "Same as UiPath Document Understanding on a queue of invoices."}\n'
    "4) Article: A GPU vendor earnings report. "
    '{"relevant": false, "reason": "Hardware finance, not a process-automation parallel.", '
    '"parallel": ""}\n'
)

AI_EXPLAINER_WITH_PARALLEL = (
    "You write a spoken explainer of this AI-news article for a listener who works "
    "with RPA and UiPath. First explain the article plainly. Then, ONLY because a "
    "genuine parallel was found, add one short section connecting the idea to RPA/"
    "UiPath in concrete terms. Do not force a comparison if it would be a stretch — "
    "the parallel is provided below.\n\n"
    f"{_PLAIN_RULES}"
)

COMPRESS_SYSTEM = (
    "Rewrite the spoken script so it is shorter, keeping every load-bearing fact. "
    "Plain spoken English only. No markdown. Target about {n} words. "
    "Keep the hook-body-recap shape. If the script greets Monish at the start, "
    "keep that greeting. Always keep a short closer so the ending still sounds finished."
)


def word_count(text: str) -> int:
    return len([w for w in (text or "").split() if w])


def build_plain_user(title: str, source: str, article_text: str, *, personal_open: bool = False) -> str:
    from narration.spoken_host import personal_open_instruction

    return (
        f"Title: {title}\nSource: {source}\n\n--- ARTICLE ---\n{article_text}\n--- END ---\n\n"
        f"{personal_open_instruction(personal_open)}\n\n"
        "Write the spoken explainer now."
    )


def build_relevance_user(title: str, article_text: str) -> str:
    excerpt = article_text[:6000]
    return f"Title: {title}\n\n--- ARTICLE ---\n{excerpt}\n--- END ---\n\nJSON:"


def build_ai_user(
    title: str,
    source: str,
    article_text: str,
    parallel: str,
    *,
    personal_open: bool = False,
) -> str:
    from narration.spoken_host import personal_open_instruction

    return (
        f"Title: {title}\nSource: {source}\n"
        f"Confirmed parallel to RPA/UiPath: {parallel}\n\n"
        f"--- ARTICLE ---\n{article_text}\n--- END ---\n\n"
        f"{personal_open_instruction(personal_open)}\n\n"
        "Write the spoken explainer now."
    )


def build_compress_user(script: str, target_words: int) -> str:
    return f"Target about {target_words} words.\n\n--- SCRIPT ---\n{script}\n--- END ---"


def is_ai_news(category: str) -> bool:
    return (category or "").strip().lower() == AI_NEWS_CATEGORY.lower()
