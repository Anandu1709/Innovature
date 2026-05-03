# Intelligent Research Assistant (Assignment 3)

RAG assistant over local PDFs with **NVIDIA NIM** chat completions, **Postgres + pgvector**, LLM routing to RAG vs **DuckDuckGo**, and **LLM-as-judge** evaluation.

## Prerequisites

1. **PostgreSQL** with the [pgvector](https://github.com/pgvector/pgvector) extension available.
2. **NVIDIA Build / NIM** API key ([NVIDIA Build](https://build.nvidia.com/)) — free tier is sufficient.

## Environment variables

Create a `.env` in this directory **manually** (do not commit it). Supported variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `NVIDIA_API_KEY` | Yes | Bearer token for NIM/OpenAI-compatible API |
| `NIM_BASE_URL` | No | Default `https://integrate.api.nvidia.com/v1` |
| `NIM_CHAT_MODEL` | No | e.g. `meta/llama-3.1-8b-instruct` (use a model enabled on your account) |
| `DATABASE_URL` | Yes | PostgreSQL URL, e.g. `postgresql://user:pass@localhost:5432/research_rag` |

Optional:

| Variable | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `5` | Chunks retrieved per query |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model (384-dim) |
| `MAX_ANSWER_TOKENS` | `800` | Cap on generator output (lower = faster) |
| `NIM_TIMEOUT` | `45` | HTTP timeout (seconds) per LLM call |
| `NIM_MAX_RETRIES` | `2` | Retries on `5xx` / `429` / timeouts |
| `NIM_BACKOFF` | `1.5` | Base backoff seconds between retries (exponential) |

## Database setup

```bash
psql "$DATABASE_URL" -f sql/schema.sql
```

## Install

```bash
cd assignment3
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## PDF corpus

Place **5–10** PDFs under `data/pdfs/`, or run:

```bash
python scripts/fetch_sample_pdfs.py
```

This downloads a small set of public arXiv PDFs for demo purposes only.

## Ingest documents

```bash
python scripts/ingest.py
```

Reads `data/pdfs/*.pdf`, chunks text (recursive split + overlap), embeds locally, inserts into Postgres.

## Manage the corpus (add / remove / sync)

Use `scripts/docs.py` to keep `data/pdfs/` and Postgres in sync:

```bash
# show what's currently ingested
python scripts/docs.py list

# add new PDFs:
#   1) drop them in data/pdfs/
#   2) sync (adds to DB, leaves existing untouched)
python scripts/docs.py sync

# remove a PDF:
#   1) delete the file from data/pdfs/  (or just from disk)
#   2) sync removes its chunks from DB automatically
python scripts/docs.py sync

# replace / re-ingest an edited PDF (keeps things consistent if it shrank)
python scripts/docs.py sync --refresh

# delete chunks for specific PDFs without touching files
python scripts/docs.py remove paper1.pdf paper2.pdf

# nuke everything (does NOT delete files on disk)
python scripts/docs.py remove --all
```

After any corpus change, just rerun `python chat.py` (or `demo_routing.py` / `evaluate_rag.py`); they always read from the current DB state.

## Routing demo

```bash
python demo_routing.py
```

Runs **two** queries: one routed to **RAG** (in-scope) and one to **DuckDuckGo** (general knowledge).

## Interactive chat

```bash
python chat.py
```

Type a question; the router decides between RAG (local PDFs) and web search.

Useful commands inside the prompt:

| Command | Effect |
|---------|--------|
| `:rag <question>` | Force the RAG path |
| `:web <question>` | Force the web search path |
| `:help` | Show the command list |
| `:exit` / `:quit` | Leave (Ctrl+C also works) |

## Evaluate RAG (LLM judge)

After ingest, ensure `eval/ground_truth.json` aligns with your ingested **`doc_hint` / contents**, then:

```bash
python eval/evaluate_rag.py
```

Scores each answer **1–5** on **Faithfulness** and **Answer Relevance**.

## Modules layout

| Path | Role |
|------|------|
| `research_assistant/nim_client.py` | Chat completions toward NIM |
| `research_assistant/generate.py` | Few-shot + CoT + strict JSON envelope |
| `research_assistant/chunking.py` | Recursive character chunking |
| `research_assistant/embeddings.py` | Local embedding encode |
| `research_assistant/db.py` | Pool + retrieval SQL |
| `research_assistant/ingest_pdf.py` | PDF text → chunks → DB |
| `research_assistant/router.py` | LLM intent → `in_scope` / `general_knowledge` |
| `research_assistant/web_search.py` | DuckDuckGo snippets |
| `research_assistant/pipeline.py` | End-to-end assistant |
