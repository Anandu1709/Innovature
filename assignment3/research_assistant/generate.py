from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from . import nim_client
from .json_utils import extract_json_object

SYSTEM_COT_JSON = """You are a careful research assistant.
You MUST think step-by-step (chain-of-thought) internally, but in your visible output you MUST return only a single JSON object (no markdown, no extra text) with exactly these keys:
- "cot": string (brief chain-of-thought reasoning steps you used)
- "summary": string (the user-facing answer; if using provided context documents, cite their file names in parentheses when stating facts from them)
- "key_entities": array of strings (important entities from the context you used)
- "confidence": number between 0 and 1 (your confidence in the answer)

Rules:
- Use only provided context when answering from context; do not invent citations.
- If context is insufficient, say so in summary and lower confidence.
- JSON must be valid UTF-8 and parseable by json.loads.
"""

FewShot = List[Dict[str, Any]]

FEW_SHOT_EXAMPLES: FewShot = [
    {
        "user": "Context:\n[docA.pdf]\nParis is the capital of France.\n\nQuestion: What is the capital of France?",
        "assistant": '{"cot":"Context explicitly states Paris is the capital of France.","summary":"Paris is the capital of France (docA.pdf).","key_entities":["Paris","France"],"confidence":0.95}',
    },
    {
        "user": "Context:\n[paper.pdf]\nThe model uses residual connections and layer normalization.\n\nQuestion: What normalization is mentioned?",
        "assistant": '{"cot":"The snippet names layer normalization.","summary":"The text mentions layer normalization (paper.pdf).","key_entities":["layer normalization","residual connections"],"confidence":0.9}',
    },
    {
        "user": "Context:\n(no relevant snippets)\n\nQuestion: Who won the Super Bowl in 2099?",
        "assistant": '{"cot":"No context provided for a future sports result; cannot answer factually.","summary":"The provided context does not contain this information.","key_entities":[],"confidence":0.2}',
    },
]


def build_messages(
    user_block: str,
    *,
    system: str = SYSTEM_COT_JSON,
    few_shot: FewShot = FEW_SHOT_EXAMPLES,
) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": system}]
    for ex in few_shot:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})
    messages.append({"role": "user", "content": user_block})
    return messages


def generate_structured_answer(
    user_block: str,
    *,
    temperature: float = 0.3,
    model: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    import os

    if max_tokens is None:
        max_tokens = int(os.getenv("MAX_ANSWER_TOKENS", "800"))
    raw = nim_client.chat_completion(
        build_messages(user_block),
        temperature=temperature,
        model=model,
        max_tokens=max_tokens,
    )
    return extract_json_object(raw)


def corpus_scope_hint(doc_names_sample: Optional[List[str]] = None) -> str:
    if not doc_names_sample:
        return "The local knowledge base contains technical PDF documents."
    preview = ", ".join(doc_names_sample[:12])
    return f"The local knowledge base covers these documents (sample): {preview}."
