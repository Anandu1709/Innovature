"""
Demonstrates router-driven behavior with two contrasting queries:
in_scope -> RAG; general_knowledge -> DuckDuckGo web snippets.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from research_assistant.pipeline import run_with_router  # noqa: E402


def main() -> None:
    examples = [
        (
            "in_scope_example",
            "According to Vaswani et al. in 'Attention Is All You Need,' what positional "
            "information is injected and what is scaled in scaled dot-product attention?",
        ),
        (
            "general_knowledge_example",
            "What is the population of Tokyo metropolitan area rounded to nearest million?",
        ),
    ]

    for label, q in examples:
        print("=" * 72)
        print(label)
        print("Q:", q)
        out = run_with_router(q)
        r = out.get("router") or {}
        print("router intent:", r.get("intent"), "| reason:", r.get("reason"))
        print("summary:", out.get("summary"))
        print("confidence:", out.get("confidence"))
        retr = out.get("retrieval") or []
        if retr:
            print("top retrieved docs:", [x["doc_name"] for x in retr[:5]])
        else:
            print("(no vector retrieval — web/general path)")
        print()


if __name__ == "__main__":
    main()
