# Module 6: Retrieval & Multimodal AI

## Overview

| Attribute | Details |
|-----------|---------|
| Module | 6 of 7 |
| Duration | ~3 Hours |
| Focus | Embeddings, Semantic Search, Vector Databases & Multimodal / Image Generation |

**Learning Objectives**

By the end of this module, you will be able to:
- Explain what embeddings are and why they turn meaning into geometry
- Choose and reason about similarity metrics (cosine, dot product, Euclidean)
- Describe the role of a vector database and pick one for a given workload
- Architect a semantic-search pipeline end-to-end: chunk → embed → index → query → rerank
- Connect retrieval to RAG and understand where it fits in an LLM application
- Explain how diffusion models turn noise into images, conditioned on text
- Compare today's image-generation platforms and call one from code
- Write effective image-generation prompts and iterate on them systematically
- Understand multimodal models (vision input, multimodal embeddings like CLIP)
- Apply Responsible-AI thinking to retrieval and generated media

---

## 1. Why Retrieval Matters

An LLM only knows two things: what was baked into its weights during training, and what you put in the prompt right now. It has **no live memory** of your documents, no access to data created after its training cut-off, and a finite context window. That gap is the entire reason retrieval exists.

| Limitation of a bare LLM | What retrieval adds |
|---|---|
| Knowledge frozen at training cut-off | Fresh, up-to-the-minute data |
| No access to your private/internal data | Your docs, tickets, wiki, codebase |
| Hallucinates when it doesn't know | Grounded answers with citations |
| Finite context window | Pull only the *relevant* slice, not everything |
| Re-training is slow and expensive | Update the index in seconds |

The core idea: **don't put everything in the prompt — put the right things.** To find "the right things" by *meaning* rather than by *exact keyword*, we need embeddings.

---

## 2. Embeddings — Turning Meaning into Geometry

An **embedding** is a list of numbers (a vector) that represents a piece of content — a word, a sentence, a paragraph, an image — as a point in a high-dimensional space. The defining property:

> **Things that mean similar things land close together; things that mean different things land far apart.**

A keyword index treats "car", "automobile", and "vehicle" as three unrelated strings. An embedding places all three near each other because they *mean* nearly the same thing. That is the leap from lexical search to **semantic** search.

### 2.1 What the Numbers Look Like

```
"The cat sat on the mat"
   → [ 0.013, -0.072, 0.401, ... , -0.118 ]   (1536 numbers)

"A feline rested on the rug"
   → [ 0.020, -0.069, 0.395, ... , -0.121 ]   (very close vector)

"Quarterly revenue grew 12%"
   → [-0.233,  0.510, 0.004, ... ,  0.298 ]   (far-away vector)
```

You never read these numbers by hand. What matters is the *geometry*: the first two vectors point in almost the same direction, the third points elsewhere.

### 2.2 How Embeddings Are Produced

A neural network (a transformer encoder) is trained so that text with similar meaning produces similar vectors. Modern embedding models are trained with **contrastive learning**: show the model pairs that *should* be close (a question and its answer) and pairs that *should* be far (unrelated text), and nudge the weights until the geometry reflects meaning.

### 2.3 OpenAI Embedding Models (current)

| Model | Dimensions | Notes |
|---|---|---|
| `text-embedding-3-small` | 1536 | Cheap, fast, strong default for most apps |
| `text-embedding-3-large` | 3072 | Highest quality, higher cost |
| `text-embedding-ada-002` | 1536 | Legacy — prefer the `-3` family |

Two important properties of the `-3` family:

- **Normalized output.** Vectors are unit length (magnitude 1), so cosine similarity and dot product give the *same* ranking — you can use the cheaper dot product.
- **Matryoshka / shortenable dimensions.** You can request fewer dimensions with the `dimensions` parameter (e.g. ask `text-embedding-3-large` for 256 dims) to save storage and speed up search, trading a little accuracy.

```python
from openai import OpenAI
client = OpenAI()

resp = client.embeddings.create(
    model="text-embedding-3-small",
    input=["The cat sat on the mat", "A feline rested on the rug"],
)
vec_a = resp.data[0].embedding   # list[float], length 1536
vec_b = resp.data[1].embedding
```

> **Cost note.** Embeddings are *much* cheaper than generation — you can embed an entire knowledge base for a few cents to a few dollars. You pay per input token, with no output tokens.

### 2.4 Token Budget for Embeddings

Embedding models have their own context limit (8191 tokens for the OpenAI `-3` family). Text longer than that must be **chunked** (see §6.1) before embedding — which is also exactly what you want for retrieval granularity.

---

## 3. Similarity — Measuring "Closeness"

Once everything is a vector, "is this relevant?" becomes "how close are these two vectors?" Three metrics dominate.

### 3.1 Cosine Similarity (the default)

Measures the **angle** between two vectors, ignoring their length. Ranges from `-1` (opposite) through `0` (unrelated/orthogonal) to `1` (identical direction).

```
cos(θ) = (A · B) / (‖A‖ · ‖B‖)
```

```python
import numpy as np

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
```

Cosine is the default because meaning is about *direction*, not magnitude — a long document and a short query about the same topic should still match.

### 3.2 Dot Product

`A · B`. When vectors are already unit-normalized (as OpenAI's are), dot product and cosine give identical rankings, and dot product is cheaper to compute. Vector databases often default to this internally for normalized data.

### 3.3 Euclidean Distance (L2)

Straight-line distance between the two points: `‖A − B‖`. *Smaller* = more similar (the opposite direction from cosine). Sensitive to magnitude, so it's less common for raw text embeddings but useful for some image and clustering tasks.

| Metric | Range | "More similar" means | Use when |
|---|---|---|---|
| Cosine | −1 … 1 | Higher | Text semantic search (default) |
| Dot product | −∞ … ∞ | Higher | Vectors are normalized; want speed |
| Euclidean (L2) | 0 … ∞ | Lower | Magnitude carries meaning; clustering |

> **Interactive:** `labs/module-06/visualizations/02-cosine-similarity.html` lets you drag two vectors and watch cosine vs. Euclidean change live.

---

## 4. Vector Databases

You *could* keep all your vectors in a Python list and compute cosine against every one on each query. That's an **exact (brute-force) search** — fine for a few thousand vectors, hopeless at millions. A **vector database** solves storage + fast search at scale.

### 4.1 What a Vector DB Gives You

- **Persistence** — vectors survive restarts; you don't re-embed every launch.
- **Approximate Nearest Neighbor (ANN) search** — sub-linear query time over millions of vectors.
- **Metadata filtering** — "find similar docs *where* `language = 'en'` and `year >= 2023`".
- **CRUD + upserts** — add/update/delete documents incrementally.
- **Scaling & ops** — sharding, replication, persistence, backups.

### 4.2 Exact vs. Approximate Search

```
Exact (brute force)            Approximate (ANN)
─────────────────────         ────────────────────────
compare query to ALL          compare to a smart SUBSET
O(n) per query                ~O(log n) per query
100% recall                   ~95–99% recall (tunable)
fine < ~10k vectors           required at 1M+ vectors
```

ANN trades a tiny bit of accuracy ("recall") for enormous speed. The dominant algorithm today is **HNSW** (Hierarchical Navigable Small World) — a layered graph you "walk" toward the nearest neighbors. Others: IVF / IVF-PQ (cluster + compress), LSH (hashing), ScaNN.

> **Interactive:** `labs/module-06/visualizations/05-vector-database-ann.html` animates an HNSW-style graph walk vs. brute-force scan.

### 4.3 The Landscape

| Database | Shape | Good fit |
|---|---|---|
| **Chroma** | Embedded / local-first | Prototyping, notebooks, this course |
| **FAISS** | Library (in-process) | Research, full control, no server |
| **pgvector** | Postgres extension | You already run Postgres; want SQL + vectors |
| **Pinecone** | Managed cloud | Zero-ops production at scale |
| **Qdrant / Weaviate / Milvus** | Self-host or cloud | Open-source production, rich filtering |
| **Redis / OpenSearch / Elasticsearch** | Existing infra + vectors | Hybrid keyword + vector in one engine |

For learning, **Chroma** is ideal: `pip install chromadb`, no server, persists to disk. We use it in `demo-03`.

```python
import chromadb
client = chromadb.PersistentClient(path="./store")
col = client.get_or_create_collection("docs")

col.add(ids=["1", "2"], documents=["hello world", "goodbye world"])
res = col.query(query_texts=["greetings"], n_results=1)
# Chroma embeds for you with a default model unless you pass your own.
```

---

## 5. Semantic Search Architecture

Putting §2–§4 together, a production semantic-search system has two phases.

```
INDEXING (offline, run when data changes)
  ┌────────┐   ┌────────┐   ┌──────────┐   ┌─────────────┐
  │ Source │──▶│ Chunk  │──▶│ Embed    │──▶│ Vector DB   │
  │ docs   │   │ + clean│   │ (model)  │   │ (+ metadata)│
  └────────┘   └────────┘   └──────────┘   └─────────────┘

QUERYING (online, per user request)
  ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐
  │ Query  │──▶│ Embed    │──▶│ ANN      │──▶│ (Rerank) │──▶│ Results │
  │ text   │   │ (SAME    │   │ search   │   │          │   │ / LLM   │
  │        │   │  model)  │   │ top-k    │   │          │   │ context │
  └────────┘   └──────────┘   └──────────┘   └──────────┘   └─────────┘
```

**The cardinal rule:** embed the query with the **exact same model** you used to embed the documents. Vectors from two different models live in incompatible spaces and their distances are meaningless.

> **Interactive:** `labs/module-06/visualizations/03-semantic-search-pipeline.html` walks a query through each stage with a toy precomputed corpus.

---

## 6. Information Retrieval with Embeddings (and the Road to RAG)

### 6.1 Chunking — the Most Underrated Step

You rarely embed whole documents. You split them into **chunks** so retrieval returns a focused, citation-sized passage rather than a 40-page PDF.

| Strategy | How | Trade-off |
|---|---|---|
| **Fixed-size** | Every N tokens (e.g. 500) | Simple; may cut mid-sentence |
| **Recursive** | Split on ¶ → sentence → word until under N | Respects structure; the common default |
| **Semantic** | Split where the topic shifts (embedding distance) | Best coherence; more compute |
| **Structural** | By headings, code blocks, table rows | Great for docs/code; format-specific |

Two dials that matter:
- **Chunk size** — too big dilutes relevance and wastes context; too small loses meaning. 200–800 tokens is typical.
- **Overlap** — repeat ~10–20% of tokens between adjacent chunks so an idea straddling a boundary isn't lost.

### 6.2 Top-k, Thresholds, and Filtering

- **top-k** — how many chunks to retrieve (commonly 3–10). More context ≠ better; irrelevant chunks distract the model.
- **Score threshold** — drop matches below a similarity floor so you return *nothing* rather than garbage.
- **Metadata filter** — combine semantic similarity with hard constraints (date, language, access permissions). This is also how you enforce **per-user authorization** — never retrieve a document the user can't see.

### 6.3 Hybrid Search & Reranking

Pure vector search can miss exact terms (product codes, names, acronyms). **Hybrid search** runs a keyword index (BM25) *and* a vector index, then fuses the rankings — commonly with **Reciprocal Rank Fusion (RRF)**.

A **reranker** (a cross-encoder such as Cohere Rerank or `bge-reranker`) then re-scores the top candidates by reading query + passage *together*, which is more accurate than comparing independent embeddings — at higher cost. Pattern: retrieve 50 cheaply, rerank to the best 5.

```
query ─┬─▶ BM25 keyword search ─┐
       │                        ├─▶ RRF fuse ─▶ rerank ─▶ top 5 ─▶ LLM
       └─▶ vector ANN search ───┘
```

### 6.4 From Retrieval to RAG

**Retrieval-Augmented Generation** = retrieval (this module) + generation (Modules 4–5). The retrieved chunks are pasted into the prompt as grounding context:

```python
context = "\n\n".join(chunk.text for chunk in top_chunks)
messages = [
    {"role": "system", "content":
        "Answer ONLY from the context. If the answer isn't there, say you don't know. Cite sources."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"},
]
```

This is the single most common enterprise LLM pattern. Module 5 §7 covered the generation half; this module is the retrieval half. `demo-03` shows the full loop with Chroma.

### 6.5 Retrieval Gotchas

- **Stale index** — re-embed when source docs change; track versions.
- **Lost in the middle** — models attend less to the middle of a long context; put the most relevant chunk first.
- **Chunk-boundary amnesia** — use overlap; consider sentence-aware splitting.
- **Garbage in** — clean boilerplate (nav bars, footers) before embedding.
- **Mismatched models** — query and docs *must* share an embedding model.

---

## 7. Introduction to Multimodal AI

So far everything has been text. **Multimodal AI** handles more than one type of data — text, images, audio, video — often in the same model.

| Direction | Example | Models |
|---|---|---|
| **Image → text** (vision) | Describe / answer questions about an image | GPT-4o, GPT-4.1, Claude, Gemini |
| **Text → image** | Generate a picture from a description | DALL·E 3, gpt-image-1, Imagen, Flux, SDXL |
| **Text/image → embedding** | Search images by text and vice versa | CLIP, SigLIP |
| **Audio ↔ text** | Transcription, text-to-speech | Whisper, TTS models |

### 7.1 Vision Input (Image Understanding)

Modern chat models accept images alongside text in the same `messages` array:

```python
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image? Any safety issues?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/site.jpg"}},
        ],
    }],
)
```

Use cases: document/receipt extraction, chart reading, accessibility alt-text, visual QA, defect detection. `demo-05` demonstrates this.

### 7.2 Multimodal Embeddings (CLIP)

**CLIP** (Contrastive Language–Image Pre-training) embeds **images and text into the *same* vector space**. A photo of a dog and the text "a dog" land near each other. This unlocks:
- **Text-to-image search** — "red sneakers" finds product photos with no manual tags.
- **Zero-shot image classification** — compare an image to label embeddings.
- **Deduplication & clustering** of image libraries.

The semantic-search architecture in §5 is identical — only the embedding model changes.

---

## 8. Text-to-Image Generation & Diffusion Models

### 8.1 The Big Idea: Denoising

A **diffusion model** learns to turn random noise into a coherent image by removing noise step by step, guided by your text prompt.

```
FORWARD (training only): add noise, step by step, until pure static
  clear image  →  ▒▓ slightly noisy  →  ▒▒▓ noisier  →  ███ pure noise

REVERSE (generation): predict & subtract noise, guided by the prompt
  ███ pure noise  →  ▒▒▓ less noise  →  ▒▓ shape emerges  →  clear image
                         ▲
              "a fox in a snowy forest, watercolor"
```

During training the model is shown images with known amounts of noise added, and learns to predict that noise. At generation time it starts from pure random noise and applies its learned denoising repeatedly (20–50 **steps**), nudged at every step toward your prompt.

> **Interactive:** `labs/module-06/visualizations/04-diffusion-denoising.html` animates noise resolving into a shape across steps.

### 8.2 Key Components

| Component | Role |
|---|---|
| **Text encoder** | Turns your prompt into vectors (CLIP / T5) |
| **U-Net / DiT** | The denoiser that predicts noise to remove at each step |
| **Cross-attention** | How the prompt *conditions* each denoising step |
| **VAE** | (Latent diffusion) compress to a small latent space → denoise there → decode. Far cheaper than pixel space |
| **Scheduler / sampler** | Controls the step schedule (DDIM, Euler, DPM++) |

**Latent diffusion** (Stable Diffusion's innovation) does the work in a compressed latent space instead of full pixels, which is why consumer GPUs can run it.

### 8.3 Important Knobs

- **Steps** — more steps = finer detail, slower. ~20–40 is a sweet spot.
- **Guidance scale (CFG)** — how strictly to follow the prompt. Too low → ignores prompt; too high → over-saturated, distorted.
- **Seed** — the starting noise. Fix it for reproducibility; change it for variations.
- **Negative prompt** (SD/Flux) — what to *avoid* ("blurry, extra fingers, watermark").
- **Aspect ratio / size** — 1:1, 16:9, portrait, etc.

### 8.4 The Platform Landscape

| Platform | Model(s) | Access | Notes |
|---|---|---|---|
| **OpenAI** | gpt-image-1, DALL·E 3 | API | `images.generate`; strong prompt adherence, text-in-image |
| **Stability AI** | SD 3.5, SDXL | API + open weights | Self-hostable, negative prompts, full control |
| **Black Forest Labs** | FLUX.1 | API + open weights | State-of-the-art open model, great quality |
| **Google** | Imagen | API (Vertex / Gemini) | Strong photorealism |
| **Midjourney** | — | Discord / web | Best-in-class aesthetics, no public API |
| **Adobe Firefly** | — | App + API | Commercially "safe" training data |

### 8.5 Calling the OpenAI Images API

```python
import base64
from openai import OpenAI
client = OpenAI()

result = client.images.generate(
    model="gpt-image-1",
    prompt="A cozy reading nook by a rainy window, warm lamp light, watercolor style",
    size="1024x1024",      # also 1536x1024 (landscape), 1024x1536 (portrait), auto
    quality="medium",      # low | medium | high
    n=1,
)
image_bytes = base64.b64decode(result.data[0].b64_json)
open("out.png", "wb").write(image_bytes)
```

`demo-04` wraps this with prompt presets and saves the file.

---

## 9. Prompting Techniques for Image Generation

Text prompting (Module 3) and image prompting are different crafts. Image models reward **descriptive, visual, comma-separated specifications**, not conversational instructions.

### 9.1 The Anatomy of a Strong Image Prompt

```
[subject] , [action/pose] , [setting/context] , [composition/shot] ,
[lighting] , [color/mood] , [medium/style] , [quality/detail cues]
```

> *"A red fox, curled asleep, in a snowy pine forest, wide establishing shot, soft golden-hour backlight, cool blue shadows, watercolor illustration, delicate paper texture, highly detailed"*

| Dimension | Example phrases |
|---|---|
| **Subject** | "an elderly fisherman", "a glass teapot" |
| **Composition / shot** | "close-up", "wide shot", "top-down", "rule of thirds" |
| **Lighting** | "golden hour", "soft studio light", "dramatic rim light", "neon" |
| **Style / medium** | "watercolor", "35mm film photo", "isometric 3D render", "line art" |
| **Mood / color** | "moody", "pastel palette", "high contrast", "muted earth tones" |
| **Detail cues** | "highly detailed", "sharp focus", "intricate" |

### 9.2 Techniques

- **Be specific, not verbose.** "Golden retriever puppy on a red checkered blanket" beats "a nice dog outside."
- **Name a style or reference medium**, not a living artist (see §10).
- **Negative prompts** (SD/Flux): list defects to avoid — `blurry, deformed hands, text, watermark, low-res`.
- **Iterate systematically.** Change *one* variable at a time (lighting, then style, then composition) so you learn what each does.
- **Fix the seed** to compare two prompts fairly; vary the seed to explore variations of a prompt you like.
- **Aspect ratio is part of the prompt** — choose it to match the composition (portrait for a person, 16:9 for a landscape).
- **Text-in-image:** gpt-image-1 renders short text well; keep it short and quote it explicitly in the prompt.

### 9.3 An Iteration Log (the disciplined way)

| Iter | Change | Result |
|---|---|---|
| 1 | Base prompt | Composition good, lighting flat |
| 2 | + "golden-hour backlight" | Lighting fixed, colors too warm |
| 3 | + "cool blue shadows" | Balanced — keep |
| 4 | seed 42 → 7 (variations) | Picked variation 7 |

> **Interactive:** `labs/module-06/visualizations/03-semantic-search-pipeline.html` (and the image-prompt builder in `demo-04`'s README) reinforce the "change one variable" habit.

---

## 10. Responsible AI for Retrieval & Generated Media

Retrieval and image generation each carry distinct risks.

**Retrieval / semantic search**
- **Access control** — filter by permission *before* returning chunks; embeddings don't respect ACLs on their own.
- **PII leakage** — the index can become a backdoor to sensitive data; scrub or gate it.
- **Provenance & citations** — always return *where* an answer came from so users can verify.
- **Stale or poisoned data** — a wrong/malicious document in the index becomes a confident wrong answer.

**Image generation**
- **Copyright & style** — avoid "in the style of [living artist]"; prefer descriptive style terms and commercially cleared models for production.
- **Deepfakes & likeness** — don't generate real, identifiable people without consent.
- **Bias** — generators inherit stereotypes from training data (occupations, beauty norms); review and diversify prompts.
- **Provenance / watermarking** — major providers embed **C2PA** content credentials; disclose AI-generated media.
- **Safety filters** — providers block sexual, violent, and hateful content; design your UX to handle refusals gracefully.

(Module 7 covers Responsible AI in depth — bias, hallucination, governance.)

---

## 11. Hands-On — Demos for This Module

All demos live under `labs/module-06/demos/`. The Python demos are `uv` projects; run each from inside its folder:

```bash
cd labs/module-06/demos/demo-01-embeddings-basics
uv sync
uv run main.py
```

| Demo | Concept | What it shows |
|---|---|---|
| `demo-01-embeddings-basics` | Embeddings + cosine | Embed sentences, rank them by similarity to a query |
| `demo-02-semantic-search` | Semantic search | In-memory top-k search over a small corpus (no DB) |
| `demo-03-vector-db-chromadb` | Vector DB + RAG | Persistent Chroma store, retrieve + answer a question |
| `demo-04-image-generation` | Text-to-image | Generate an image via the OpenAI Images API |
| `demo-05-vision-multimodal` | Vision input | Ask a multimodal model questions about an image |

**Interactive HTML visualizations** (open directly in a browser — no install):

```bash
open labs/module-06/visualizations/index.html
```

| File | Visualizes |
|---|---|
| `01-embeddings-vector-space.html` | Words as points; similar meanings cluster |
| `02-cosine-similarity.html` | Drag two vectors; cosine vs. Euclidean live |
| `03-semantic-search-pipeline.html` | A query flowing through chunk→embed→search→rerank |
| `04-diffusion-denoising.html` | Noise resolving into an image across steps |
| `05-vector-database-ann.html` | Brute-force scan vs. ANN graph walk |

> **API keys.** The Python demos need an `OPENAI_API_KEY`. Copy each demo's `.envbackup` to `.env` and fill it in. Never commit `.env`.

---

## 12. Recap & What's Next

- **Embeddings** turn meaning into geometry; similar content → nearby vectors.
- **Similarity** is measured by angle (cosine, the default) or distance (Euclidean).
- **Vector databases** store vectors and do fast **approximate** nearest-neighbor search at scale.
- **Semantic search** = chunk → embed → index → embed query → ANN → (rerank). Same model on both sides.
- **Retrieval feeds RAG**, the dominant pattern for grounding LLMs on private/fresh data.
- **Diffusion models** generate images by denoising random static, conditioned on text.
- **Image prompting** is a visual specification craft — describe subject, composition, lighting, style; iterate one variable at a time.
- **Multimodal models** read images (vision) and share a space across modalities (CLIP).
- **Responsibly**: enforce access control on retrieval; disclose and de-bias generated media.

**Next — Module 7: Responsible AI & Mini Project.** You'll study bias, hallucination, privacy, and governance, then build and demo a capstone AI application.

---

## Appendix A — Glossary

| Term | Meaning |
|---|---|
| **Embedding** | A vector representing the meaning of content |
| **Vector** | An ordered list of numbers; a point in N-dimensional space |
| **Dimension** | The length of a vector (e.g. 1536) |
| **Cosine similarity** | Angle-based closeness, −1…1 (default for text) |
| **Dot product** | Sum of element-wise products; equals cosine for unit vectors |
| **Euclidean (L2)** | Straight-line distance between two points |
| **ANN** | Approximate Nearest Neighbor — fast, slightly-inexact search |
| **HNSW** | Hierarchical Navigable Small World — graph-based ANN index |
| **Vector database** | Store + ANN search + metadata filtering for vectors |
| **Chunk** | A small slice of a document that gets embedded |
| **Overlap** | Shared tokens between adjacent chunks |
| **top-k** | Number of results returned by a search |
| **RAG** | Retrieval-Augmented Generation — retrieve then generate |
| **Reranker** | Cross-encoder that re-scores candidates by reading query+passage together |
| **BM25** | Classic keyword-relevance scoring algorithm |
| **Hybrid search** | Combining keyword (BM25) and vector search |
| **RRF** | Reciprocal Rank Fusion — merges multiple rankings |
| **Multimodal** | Handling more than one data type (text, image, audio…) |
| **CLIP** | Model embedding images and text into one shared space |
| **Diffusion model** | Generates images by iteratively denoising random noise |
| **Latent diffusion** | Diffusion performed in a compressed latent space |
| **U-Net / DiT** | The denoising network inside a diffusion model |
| **CFG (guidance scale)** | How strongly generation follows the prompt |
| **Seed** | The random starting point; fixes/varies the output |
| **Negative prompt** | What the image model should avoid |
| **C2PA** | Content provenance standard for labeling AI media |

---

## Appendix B — Further Reading

- OpenAI — Embeddings guide and the `text-embedding-3` model family
- OpenAI — Images (gpt-image-1) and Vision guides
- Chroma, Pinecone, Qdrant, Weaviate, pgvector — vector DB docs
- "Attention Is All You Need" (transformers) and the CLIP paper (Radford et al.)
- "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., Stable Diffusion)
- Cohere Rerank / `bge-reranker` — reranking for retrieval
- C2PA — Content Credentials specification

---

### Labs for this Module

- `labs/module-06/demos/demo-01-embeddings-basics` — embeddings & similarity
- `labs/module-06/demos/demo-02-semantic-search` — in-memory semantic search
- `labs/module-06/demos/demo-03-vector-db-chromadb` — vector DB + RAG
- `labs/module-06/demos/demo-04-image-generation` — text-to-image
- `labs/module-06/demos/demo-05-vision-multimodal` — image understanding
- `labs/module-06/visualizations/` — 5 interactive HTML visualizations
