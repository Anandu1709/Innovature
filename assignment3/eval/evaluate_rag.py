"""LLM-as-judge evaluation for the RAG path (Faithfulness & Answer Relevance, 1-5)."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from research_assistant import nim_client  # noqa: E402
from research_assistant.json_utils import extract_json_object  # noqa: E402
from research_assistant.pipeline import rag_only  # noqa: E402


JUDGE_SYSTEM = """You are an impartial evaluator for a Retrieval-Augmented Generation system.
Given the USER QUESTION, REFERENCE/GROUND TRUTH ANSWER, RETRIEVAL CONTEXT (chunks used), and MODEL OUTPUT SUMMARY,

Score BOTH:
1) faithfulness_to_context (integer 1-5): is the MODEL OUTPUT strictly supported by RETRIEVAL CONTEXT (not contradicted)? Penalize invented facts absent from context.
2) answer_relevance (integer 1-5): does the MODEL OUTPUT directly answer the USER QUESTION?

Respond ONLY with JSON:
{\"faithfulness_to_context\": <1-5>, \"answer_relevance\": <1-5>, \"faithfulness_notes\": \"...\", \"relevance_notes\": \"...\"}
"""


def judge_one(
    question: str,
    reference: str,
    summary: str,
    context_blob: str,
) -> dict:
    user = json.dumps(
        {
            "user_question": question,
            "ground_truth_answer": reference,
            "retrieval_context": context_blob,
            "model_summary_to_grade": summary,
        },
        ensure_ascii=False,
    )
    raw = nim_client.chat_completion(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return extract_json_object(raw)


def main() -> None:
    gt_path = Path(__file__).with_name("ground_truth.json")
    items = json.loads(gt_path.read_text(encoding="utf-8"))

    faith_scores = []
    rel_scores = []

    print("Running RAG + judge over", len(items), "questions\n")

    for item in items:
        qid = item["id"]
        q = item["question"]
        ref = item["reference_answer"]

        rag_out = rag_only(q)
        summary = rag_out.get("summary", "") or ""
        retr = rag_out.get("retrieval") or []
        ctx_lines = []
        for r in retr:
            ctx_lines.append(f"[{r['doc_name']}]\n{r['text']}")
        ctx_blob = "\n\n".join(ctx_lines)

        scores = judge_one(q, ref, summary, ctx_blob)
        fs = int(scores.get("faithfulness_to_context", 0))
        ar = int(scores.get("answer_relevance", 0))

        faith_scores.append(fs)
        rel_scores.append(ar)

        print("-" * 72)
        print("id:", qid)
        print("faithfulness:", fs, "| answer_relevance:", ar)
        print("faithfulness_notes:", scores.get("faithfulness_notes"))
        print("relevance_notes:", scores.get("relevance_notes"))

    n = len(items)
    if n:
        print("\nMeans — faithfulness:", sum(faith_scores) / n, "| relevance:", sum(rel_scores) / n)


if __name__ == "__main__":
    main()
