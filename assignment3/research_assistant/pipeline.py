"""End-to-end assistant: retrieve, generate, route."""

import os
from typing import Any, Dict, List, Optional

from . import db, embeddings, generate, router, web_search


def _format_context_block(chunks: List[dict]) -> str:
    lines: List[str] = []
    for c in chunks:
        lines.append(f"[{c['doc_name']}]")
        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines).strip()


def rag_answer(
    user_query: str,
    *,
    top_k: Optional[int] = None,
    temperature: float = 0.25,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    k = top_k or int(os.getenv("TOP_K", "5"))
    qvec = embeddings.encode_texts([user_query])[0]
    retrieved = db.retrieve_top_k(qvec, k)
    ctx = _format_context_block(retrieved)
    user_block = (
        "Use the CONTEXT below. When stating facts derived from CONTEXT, cite the "
        "source file names shown in brackets like (filename.pdf).\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"Question: {user_query}"
    )
    structured = generate.generate_structured_answer(
        user_block, temperature=temperature, model=model
    )
    return {
        **structured,
        "retrieval": retrieved,
        "route": {"intent": "in_scope"},
    }


def web_answer(
    user_query: str,
    *,
    temperature: float = 0.35,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    web_ctx = web_search.duckduckgo_snippets(user_query)
    user_block = (
        "Answer using WEB SNIPPETS. These are external search results—not local PDFs. "
        "Do NOT fabricate citations to PDF filenames.\n\n"
        f"WEB SNIPPETS:\n{web_ctx}\n\n"
        f"Question: {user_query}"
    )
    structured = generate.generate_structured_answer(
        user_block, temperature=temperature, model=model
    )
    return {**structured, "retrieval": [], "route": {"intent": "general_knowledge"}}


def run_with_router(
    user_query: str,
    *,
    doc_names_sample: Optional[List[str]] = None,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    # The caller may pass a cached corpus index to avoid an extra DB roundtrip.
    if doc_names_sample is None:
        doc_names_sample = db.list_distinct_doc_names()
    decision = router.route_query(
        user_query, doc_names_sample=doc_names_sample, model=model
    )
    if decision["intent"] == "general_knowledge":
        out = web_answer(user_query, model=model)
    else:
        out = rag_answer(user_query, top_k=top_k, model=model)
    out["router"] = decision
    return out


def rag_only(
    user_query: str,
    *,
    top_k: Optional[int] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Used by evaluation to grade the RAG path only."""
    return rag_answer(user_query, top_k=top_k, model=model)
