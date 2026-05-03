"""LLM Router: classify in_scope (use RAG on local PDF corpus) vs general_knowledge (web search)."""

from typing import Dict, Optional

from . import nim_client
from .json_utils import extract_json_object
from .generate import corpus_scope_hint


ROUTER_SYSTEM = """You classify user queries for an assistant backed by LOCAL PDF DOCUMENTS ONLY.
Respond with ONLY a JSON object with keys:
- "intent": either "in_scope" or "general_knowledge"
- "reason": short string

Definitions:
- in_scope: the user likely expects an answer grounded in research/technical PDFs in the corpus (methods, architectures, equations, definitions, paper specifics, summaries of uploaded docs).
- general_knowledge: broad world facts, current events, pop culture, or questions clearly not relying on technical PDF uploads.

{scope_hint}
"""


def route_query(
    user_query: str,
    *,
    doc_names_sample: Optional[list] = None,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> Dict[str, str]:
    scope = corpus_scope_hint(doc_names_sample)
    system = ROUTER_SYSTEM.format(scope_hint=scope)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": 'Examples:\nQ: Summarize multi-head attention from our papers.\n{"intent":"in_scope","reason":"asks about papers in corpus topic"}\n\nQ: What is today\'s temperature in Kyoto?\n{"intent":"general_knowledge","reason":"weather/real-world current facts"}\n\nQ: ' + user_query,
        },
    ]
    raw = nim_client.chat_completion(
        messages,
        temperature=temperature,
        model=model,
        max_tokens=200,
    )
    parsed = extract_json_object(raw)
    intent = parsed.get("intent", "in_scope")
    if intent not in ("in_scope", "general_knowledge"):
        intent = "in_scope"
    reason = str(parsed.get("reason", ""))
    return {"intent": intent, "reason": reason}
