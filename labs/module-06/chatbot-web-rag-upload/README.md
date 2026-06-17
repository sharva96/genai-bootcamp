# Demo 06 — RAG Chatbot (Upload Edition)

The **RAG chatbot from `chatbot-web-rag`, but you build the knowledge base
yourself** — drag in your own `.txt`, `.md`, or `.pdf` files and ask questions
about them. Same retrieval pipeline (chunk → embed → Chroma → retrieve →
ground), now driven by **runtime uploads** instead of a fixed `data/` folder.

> Companion to **Guide 06 — Retrieval & Multimodal AI**, §4 (Vector Databases)
> and §6.4 (From Retrieval to RAG). This variant is the "bring your own
> documents" version of the capstone.

---

## What's new vs. `chatbot-web-rag`

| `chatbot-web-rag` | This demo |
|---|---|
| Fixed knowledge base baked into `data/*.md` | **Upload panel** — index `.txt`, `.md`, `.pdf` from the browser |
| One-time build on first question | **Add / remove / clear** documents live; re-uploading a file replaces it |
| Markdown-heading chunking only | Heading chunks for `.md`, **overlapping windows** for `.txt` & PDF text |
| 5 personas | Trimmed to **Sage** (RAG) + **Nova** (no RAG) to keep the demo focused |

## How it works

```
upload .txt/.md/.pdf ──extract text──▶ chunk ──▶ embed (OpenAI) ──▶ Chroma collection
                                                                          │
        user question ──embed──▶ top-k nearest ─────────────────────────┘──▶ inject as context ──▶ LLM answers
```

- `app/rag.py` — `extract_text()` reads each format (PDF via `pypdf`),
  `chunk_document()` splits it (headings for `.md`, ~1200-char overlapping
  windows otherwise), then `index_document()` embeds the chunks and adds them to
  a persistent Chroma collection. `list_documents()` / `remove_document()` /
  `clear_all()` manage the store.
- `app/main.py` — adds upload endpoints; the chat turn is unchanged from the
  original (retrieve top-k → stream a `sources` event → inject as a system
  message → answer only from that context).
- Everything stays **lazy**: nothing hits OpenAI or Chroma until you upload or
  ask, so the app imports (and the tests run) without an API key.

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- OpenAI API key with access to `gpt-4o-mini` and `text-embedding-3-small`

## Setup

```bash
# from labs/module-06/chatbot-web-rag-upload/

uv sync

cp .env.example .env
# Edit .env — set OPENAI_API_KEY
```

> Never commit `.env`. It is already in `.gitignore`.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000>.

## Try it (the 60-second demo)

1. **Build the index.** In the **Knowledge base** panel, either drop a file in
   the upload zone or click **Load samples** (indexes the bundled TimeFlow docs,
   including a `.txt`). Watch the chunk count climb.
2. **Ask a grounded question** — *"How long does the battery last?"* The
   **Retrieved context** panel fills with the chunks that were pulled (with
   similarity scores) and the reply cites them as `[n]`.
3. **Ask something not in the docs** — *"What's the stock price?"* Retrieval
   still runs, but Sage honestly says it doesn't know.
4. **Upload your own file** — drag in any `.pdf`, `.md`, or `.txt` and ask about
   it immediately.
5. **Switch to Nova** (no RAG pill) — it can't see your documents; it answers
   from the model directly. Switch back to feel the difference.
6. **Clear all** wipes the knowledge base for a fresh run.

## Project Layout

```
chatbot-web-rag-upload/
├── .env.example                  # API key + model template
├── pyproject.toml                # adds pypdf + python-multipart
├── data/                         # sample docs for the "Load samples" button
│   ├── timeflow-x1-specs.md
│   ├── timeflow-app.md
│   ├── warranty-and-returns.md
│   ├── accessories.md
│   └── timeflow-quicktips.txt    # plain-text sample (window chunking)
├── app/
│   ├── main.py                   # FastAPI app + streaming + upload endpoints
│   ├── rag.py                    # extract → chunk → embed → Chroma → retrieve
│   ├── personas.py               # Sage (RAG) + Nova (no RAG)
│   ├── memory.py                 # Sliding/summary/full strategies (unchanged)
│   ├── sessions.py               # In-memory session store (unchanged)
│   └── guardrails.py             # Input + output filters (unchanged)
├── tests/
│   └── test_smoke.py             # wiring + extraction/chunking tests (no key)
└── static/
    └── index.html                # UI with upload panel + Retrieved-context panel
```

## API Reference

Document endpoints (new in this edition):

| method & path | purpose |
|---|---|
| `GET /api/documents`          | list indexed files + per-file chunk counts |
| `POST /api/documents`         | upload + index one or more files (multipart `files`) |
| `POST /api/documents/samples` | index the bundled `data/` samples |
| `DELETE /api/documents`       | clear the whole knowledge base |
| `DELETE /api/documents/{src}` | remove one file from the index |

The streaming `chat` endpoint emits the same Server-Sent Events as
`chatbot-web-rag` (`sources`, `meta`, `delta`, `guardrail`, `done`, `error`).

## Test

```bash
uv run pytest
```

The smoke tests cover routing, session lifecycle, persona/RAG metadata, the
input guardrail, upload validation, and the text-extraction + chunking logic —
all without an API key.

## Troubleshooting

**`AuthenticationError`** — Set `OPENAI_API_KEY` in `.env`.

**Upload says "no readable text found"** — Scanned/image-only PDFs have no
extractable text layer. Use a text-based PDF, or paste the content into a `.txt`.

**Stale answers after editing a file** — Re-upload it (same name replaces the old
version), or use **Clear all** and rebuild.

**`ModuleNotFoundError: app`** — Run from inside `chatbot-web-rag-upload/`.

**`uvicorn: command not found`** — Use `uv run uvicorn ...` (not bare `uvicorn`).

## Beyond This Demo

- **Token-aware chunking** — split on token counts with tuned overlap instead of
  character windows.
- **More formats** — add `.docx` (python-docx), `.html`, or OCR for scanned PDFs.
- **Per-document collections** — isolate each upload, or scope retrieval by file.
- **Hybrid retrieval** — combine vector search with keyword (BM25) re-ranking.
- **Swap the store** — the pipeline shape is identical for `pgvector` or Qdrant.
