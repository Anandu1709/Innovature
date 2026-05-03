"""
Interactive command-line chat for the Intelligent Research Assistant.

Type a question; the LLM router decides between RAG (local PDFs) and a web search,
and the answer (with citations or web sources) is printed.

Commands:
  :rag <question>     force the RAG path
  :web <question>     force the web search path
  :help               show commands
  :exit / :quit       leave (Ctrl+C also works)
"""

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from research_assistant import db, embeddings  # noqa: E402
from research_assistant.pipeline import (  # noqa: E402
    rag_answer,
    run_with_router,
    web_answer,
)


HELP_TEXT = """\
Commands:
  :rag <question>   route to RAG (local PDFs) regardless of router
  :web <question>   route to web search regardless of router
  :help             show this help
  :exit / :quit     quit
Anything else is sent through the router.
"""


# Cache the doc-name list once per session so the router prompt doesn't
# requery Postgres for every turn.
_DOC_NAMES_CACHE: list[str] | None = None


def _doc_names_sample() -> list[str]:
    global _DOC_NAMES_CACHE
    if _DOC_NAMES_CACHE is None:
        try:
            _DOC_NAMES_CACHE = db.list_distinct_doc_names()
        except Exception:
            _DOC_NAMES_CACHE = []
    return _DOC_NAMES_CACHE


class _Spinner:
    """Tiny stderr spinner so the user sees progress while NIM is working."""

    def __init__(self, label: str = "thinking") -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Spinner":
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        sys.stderr.write("\r" + " " * 40 + "\r")
        sys.stderr.flush()

    def _spin(self) -> None:
        frames = "|/-\\"
        i = 0
        t0 = time.time()
        while not self._stop.is_set():
            elapsed = time.time() - t0
            sys.stderr.write(f"\r{self.label} {frames[i % 4]} ({elapsed:4.1f}s)")
            sys.stderr.flush()
            i += 1
            time.sleep(0.1)


def _print_answer(out: Dict[str, Any]) -> None:
    router_info = out.get("router") or out.get("route") or {}
    intent = router_info.get("intent", "?")
    reason = router_info.get("reason")
    print(f"\n[route] intent={intent}" + (f" | reason={reason}" if reason else ""))

    summary = out.get("summary") or "(no summary)"
    print("\nAnswer:")
    print(summary)

    entities = out.get("key_entities") or []
    if entities:
        print("\nKey entities:", ", ".join(map(str, entities)))

    confidence = out.get("confidence")
    if confidence is not None:
        print(f"Confidence: {confidence}")

    retrieved = out.get("retrieval") or []
    if retrieved:
        seen = []
        for r in retrieved:
            name = r.get("doc_name")
            if name and name not in seen:
                seen.append(name)
        print("Sources (PDF):", ", ".join(seen))


def _handle(line: str) -> bool:
    """Return False to exit the loop."""
    s = line.strip()
    if not s:
        return True

    low = s.lower()
    if low in {":exit", ":quit", "exit", "quit"}:
        return False
    if low in {":help", "help", "?"}:
        print(HELP_TEXT)
        return True

    try:
        with _Spinner():
            if s.startswith(":rag "):
                q = s[len(":rag "):].strip()
                if not q:
                    print("Provide a question after :rag")
                    return True
                out = rag_answer(q)
                out["router"] = {"intent": "in_scope", "reason": "forced :rag"}
            elif s.startswith(":web "):
                q = s[len(":web "):].strip()
                if not q:
                    print("Provide a question after :web")
                    return True
                out = web_answer(q)
                out["router"] = {"intent": "general_knowledge", "reason": "forced :web"}
            else:
                out = run_with_router(s, doc_names_sample=_doc_names_sample())
        _print_answer(out)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"\n[error] {type(exc).__name__}: {exc}")
    return True


def _prewarm() -> None:
    """Load the embedding model and cache doc-name list to remove cold-start latency."""
    sys.stderr.write("warming up embedding model and corpus index... ")
    sys.stderr.flush()
    try:
        embeddings.get_embedding_model()
        _doc_names_sample()
        sys.stderr.write("done.\n")
    except Exception as exc:
        sys.stderr.write(f"skipped ({type(exc).__name__}: {exc})\n")


def main() -> None:
    print("Intelligent Research Assistant — type :help for commands, :exit to quit.")
    _prewarm()
    print()
    try:
        while True:
            try:
                line = input("you> ")
            except EOFError:
                print()
                return
            if not _handle(line):
                return
            print()
    except KeyboardInterrupt:
        print("\nbye.")


if __name__ == "__main__":
    main()
