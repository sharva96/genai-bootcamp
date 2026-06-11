# Module 2: NLP, Transformer Architecture & LLM Fundamentals

## Overview

| Attribute | Details |
|-----------|---------|
| Module | 2 of 7 |
| Duration | ~3 Hours |
| Focus | Technical Foundations of Language AI |

**Learning Objectives**

By the end of this module, you will be able to:
- Define NLP and describe its core tasks and real-world applications
- Trace the evolution of NLP from rule-based systems to LLMs
- Explain how text is tokenized and represented as numbers
- Understand what embeddings are and why they matter
- Describe the Transformer architecture and the role of attention
- Distinguish between encoder, decoder, and encoder-decoder models
- Explain how LLMs work, including context windows and the prompt-response lifecycle

---

## 1. Introduction to Natural Language Processing (NLP)

**Natural Language Processing (NLP)** is a branch of AI that enables computers to understand, interpret, manipulate, and generate human language — in text or speech form.

Language is the hardest form of data for computers to handle because it is:
- **Ambiguous** — "I saw the man with the telescope" (who has the telescope?)
- **Context-dependent** — "bank" means different things in different sentences
- **Implicit** — Much meaning is unstated and assumed
- **Ever-evolving** — Slang, new words, shifting usage over time

NLP bridges the gap between human communication and machine computation.

### NLP vs Computational Linguistics vs Speech Processing

| Field | Focus |
|-------|-------|
| **NLP** | Text understanding and generation; practical tasks |
| **Computational Linguistics** | Formal study of language structure using computation |
| **Speech Processing** | Audio signal → text (ASR) and text → audio (TTS) |

Modern LLMs blur these boundaries — GPT-4o, for example, handles all three natively.

---

## 2. NLP Tasks and Applications

NLP is not one task but a **family of tasks** ranging from low-level text processing to high-level reasoning.

### Core NLP Task Taxonomy

```
NLP Tasks
├── Understanding (Analysis)
│   ├── Text Classification
│   ├── Named Entity Recognition (NER)
│   ├── Part-of-Speech (POS) Tagging
│   ├── Dependency Parsing
│   ├── Coreference Resolution
│   ├── Sentiment Analysis
│   └── Information Extraction
│
├── Generation
│   ├── Text Summarization
│   ├── Machine Translation
│   ├── Question Answering
│   ├── Dialogue / Conversation
│   └── Text Generation
│
└── Retrieval
    ├── Semantic Search
    ├── Document Ranking
    └── Fact Verification
```

### Task Reference Table

| Task | Description | Example | Business Use |
|------|-------------|---------|-------------|
| **Text Classification** | Assign a label to a document | Spam / Not Spam | Email filtering, support ticket routing |
| **Sentiment Analysis** | Detect emotional tone | Positive / Negative / Neutral | Customer feedback analysis, brand monitoring |
| **Named Entity Recognition** | Extract people, places, orgs, dates | "Apple [ORG] CEO Tim Cook [PER]" | Contract analysis, news monitoring |
| **Machine Translation** | Convert between languages | EN → FR | Localization, global support |
| **Summarization** | Condense a document | "3-sentence summary of this 20-page report" | Research, legal, financial document processing |
| **Question Answering** | Answer questions from a context | RAG-based enterprise search | Knowledge bases, internal search |
| **Text Generation** | Produce new text | Draft an email | Copilots, content generation |
| **Coreference Resolution** | Identify who "he", "she", "it" refers to | "John said he was tired" → John | Conversation understanding, summarization |
| **POS Tagging** | Label grammatical role of each word | "runs" → Verb | Parsing, grammar correction |
| **Dependency Parsing** | Map grammatical relationships | Subject → Verb → Object tree | Structured data extraction |

---

## 3. Evolution of NLP Systems

NLP has gone through four distinct paradigms over 70 years.

### Era 1: Rule-Based NLP (1950s–1980s)

Linguists hand-crafted **grammars, dictionaries, and rules** to parse and process language.

- **ELIZA (1966)** — First chatbot; pattern-matched inputs to scripted responses; users anthropomorphized it
- **Parsing systems** — Context-free grammars, parse trees
- **Limitations:** Brittle; couldn't generalize; maintaining rules was expensive; failed on unseen inputs

### Era 2: Statistical NLP (1990s–2000s)

Shifted from hand-crafted rules to **learning patterns from corpora** (large text collections).

Key models:
- **N-gram language models** — Probability of next word given previous N words
- **Hidden Markov Models (HMM)** — For POS tagging and speech recognition
- **Naive Bayes / SVM** — For text classification

```
P("the cat sat on the mat") = P("sat"|"cat") × P("on"|"sat") × ...
```

- **Limitation:** N-gram models have no semantic understanding; "Paris" and "France" are unrelated unless seen together often

### Era 3: Neural NLP (2013–2017)

**Word2Vec (2013, Google)** was the breakthrough — words could be represented as dense vectors that captured semantic meaning.

```
king - man + woman ≈ queen    (vector arithmetic on meaning)
Paris - France + Germany ≈ Berlin
```

- **Word2Vec, GloVe** — Static word embeddings (one vector per word)
- **RNNs / LSTMs** — Process sequences; maintain "memory" across tokens
- **Seq2Seq models** — Encoder-decoder for translation and summarization
- **Limitation:** RNNs process tokens one-by-one (slow, hard to parallelize); long-range dependencies are lost

### Era 4: Transformer Era (2017–present)

**"Attention Is All You Need"** (Vaswani et al., 2017, Google Brain) — replaced recurrence with the **self-attention mechanism**, enabling:
- Full parallelization during training
- Direct modeling of relationships between any two tokens regardless of distance
- Scaling to billions of parameters

**Milestones:**
| Year | Model | Significance |
|------|-------|-------------|
| 2017 | Transformer | Foundational architecture |
| 2018 | BERT (Google) | Bidirectional pre-training; fine-tuning paradigm |
| 2018 | GPT-1 (OpenAI) | Generative pre-training on decoder |
| 2019 | GPT-2 | Scaled up; "too dangerous to release" (initially) |
| 2020 | GPT-3 (175B) | Few-shot learning; emergent capabilities |
| 2022 | ChatGPT | RLHF alignment; mainstream adoption |
| 2023 | GPT-4, Claude 2, Gemini | Multimodal, long context |
| 2024–2025 | GPT-4o, Claude 3/4, Gemini 1.5/2 | Real-time multimodal, 1M+ context |

---

## 4. Tokens and Language Representation

Computers can only process numbers. The process of converting text to numbers is called **tokenization**.

### What is a Token?

A **token** is the basic unit of text that a model processes. It is NOT the same as a word.

| Text | Tokens |
|------|--------|
| "Hello, world!" | `["Hello", ",", " world", "!"]` — 4 tokens |
| "unbelievable" | `["un", "believ", "able"]` — 3 tokens |
| "ChatGPT" | `["Chat", "G", "PT"]` — 3 tokens |
| "2024" | `["2024"]` — 1 token |

**Rule of thumb:** 1 token ≈ 4 characters ≈ ¾ of a word (for English)
- 100 tokens ≈ 75 words
- 1,000 tokens ≈ 750 words ≈ ~1.5 pages

### Tokenization Algorithms

#### Byte-Pair Encoding (BPE)
The most common approach (used by GPT models):
1. Start with individual characters as the vocabulary
2. Repeatedly merge the most frequent adjacent pair into a new token
3. Stop when vocabulary size is reached (typically 50,000–100,000 tokens)

**Why BPE?**
- Handles unknown words gracefully (rare words split into sub-word pieces)
- Fixed vocabulary size
- Works across languages and code

#### WordPiece
Used by BERT; similar to BPE but merges based on likelihood rather than raw frequency.

#### SentencePiece
Language-agnostic tokenizer used by models like T5, Llama.

### Vocabulary and Token IDs

Each token is mapped to a unique integer ID:
```
"Hello" → 15496
"world" → 995
"!"     → 0
```

This is what the model actually processes — sequences of integers.

### Why Tokenization Matters for Practitioners

1. **Cost** — API pricing is per token (input + output)
2. **Context limits** — Model max context is measured in tokens, not words
3. **Counting** — A 10-page document might be ~7,500 tokens
4. **Non-English text** — Languages like Chinese, Arabic use more tokens per word than English, increasing costs

```python
# Check token counts using tiktoken (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")
text = "The quick brown fox jumps over the lazy dog."
tokens = enc.encode(text)
print(f"Token count: {len(tokens)}")   # 10 tokens
print(f"Token IDs: {tokens}")
print(f"Decoded: {[enc.decode([t]) for t in tokens]}")
```

---

## 5. Embeddings Fundamentals

### The Problem: Text Has No Natural Numerical Form

Words cannot be compared mathematically using raw integers (token IDs).
- Token 100 is not "twice" token 50
- "King" and "Queen" have unrelated IDs but related meanings

### The Solution: Embeddings

An **embedding** is a dense vector of real numbers that represents the **semantic meaning** of a token, word, sentence, or document in a high-dimensional space.

```
"king"   → [0.32, -0.15, 0.87, 0.04, ..., -0.22]   # 768 or 1536 dimensions
"queen"  → [0.30, -0.13, 0.85, 0.06, ..., -0.20]   # Very similar vector
"pizza"  → [-0.91, 0.44, -0.02, 0.77, ..., 0.11]   # Very different vector
```

### Key Properties of Good Embeddings

1. **Semantic similarity** — Similar-meaning words have similar vectors (small distance between them)
2. **Analogical reasoning** — Vector arithmetic captures relationships
   ```
   embed("king") - embed("man") + embed("woman") ≈ embed("queen")
   ```
3. **Geometry encodes meaning** — Clusters form around topics; directions encode properties

### Types of Embeddings

| Type | Scope | Example Models | Use Case |
|------|-------|---------------|---------|
| **Word embeddings** | Single word | Word2Vec, GloVe | Foundational; mostly historical |
| **Contextual embeddings** | Word in context | BERT hidden states | "bank" near "river" ≠ "bank" near "money" |
| **Sentence embeddings** | Full sentence | SBERT, OpenAI text-embedding-3 | Semantic search, clustering |
| **Document embeddings** | Full document | Longformer, doc2vec | Document similarity, retrieval |

### Measuring Similarity: Cosine Similarity

The standard way to compare two embeddings:

```
cosine_similarity(A, B) = (A · B) / (|A| × |B|)
```

- Result ranges from -1 (opposite) to 1 (identical)
- Values > 0.8 generally indicate high semantic similarity

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

e1 = get_embedding("The dog chased the cat")
e2 = get_embedding("A puppy ran after a kitten")
e3 = get_embedding("The stock market crashed today")

print(f"Sentence 1 vs 2: {cosine_similarity(e1, e2):.4f}")  # High similarity
print(f"Sentence 1 vs 3: {cosine_similarity(e1, e3):.4f}")  # Low similarity
```

### Embeddings in Practice

Embeddings are the engine behind:
- **Semantic search** — "Find documents similar to this query" (beyond keyword matching)
- **Recommendation engines** — "Users who liked X also liked Y"
- **Clustering** — Group documents by topic automatically
- **Retrieval-Augmented Generation (RAG)** — Find relevant context to inject into prompts

---

## 6. Challenges of Traditional NLP Models

Understanding why older approaches failed informs why Transformers succeeded.

### Problems with Rule-Based Systems
- **Brittleness** — Any sentence outside the defined grammar fails
- **Maintenance overhead** — Rules multiply as edge cases are discovered
- **No statistical grounding** — Can't rank outputs by likelihood

### Problems with N-gram Models
- **No semantic understanding** — "good" and "excellent" are unrelated unless co-occurring
- **Data sparsity** — Most N-gram combinations never appear in training data
- **Context limitation** — Trigrams can only "see" 2 words back

### Problems with RNNs and LSTMs
- **Vanishing gradients** — Gradients shrink exponentially during backpropagation through many steps; long-range dependencies are lost
- **Sequential processing** — Cannot parallelize across time steps; training is slow
- **Fixed hidden state** — The entire "memory" of the sequence must compress into one vector

```
RNN bottleneck:
[word1] → [word2] → [word3] → ... → [word500] → [hidden state] → output
                                                       ↑
                              Everything must pass through this bottleneck
```

### Summary: Why Transformers Won

| Challenge | RNN/LSTM | Transformer |
|-----------|----------|-------------|
| Long-range dependencies | Poor (vanishing gradient) | Excellent (direct attention) |
| Parallelization | None (sequential) | Full (attention is matrix ops) |
| Training speed | Slow | Fast (GPU-optimized) |
| Scaling | Difficult | Excellent — scales with data + compute |

---

## 7. Introduction to Transformer Architecture

The Transformer is the architecture underlying virtually every modern LLM. Published in **"Attention Is All You Need"** (2017), it replaced recurrence entirely with attention.

### High-Level Architecture

```
                    ┌─────────────────────────────┐
                    │         OUTPUT               │
                    │   (Predicted next token)     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       Linear + Softmax       │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Decoder Stack (N×)       │
                    │  ┌─────────────────────────┐ │
                    │  │  Self-Attention          │ │
                    │  │  Cross-Attention         │ │
                    │  │  Feed-Forward Network    │ │
                    │  │  Layer Normalization     │ │
                    │  └─────────────────────────┘ │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Encoder Stack (N×)       │
                    │  ┌─────────────────────────┐ │
                    │  │  Self-Attention          │ │
                    │  │  Feed-Forward Network    │ │
                    │  │  Layer Normalization     │ │
                    │  └─────────────────────────┘ │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Input Embeddings +         │
                    │   Positional Encoding        │
                    └──────────────┬──────────────┘
                                   │
                              [Input Text]
```

### Core Components

#### Input Embeddings
Convert token IDs into dense embedding vectors (dimension typically 768–12,288 depending on model size).

#### Positional Encoding
Transformers process all tokens in parallel — they have no inherent sense of order. Positional encodings add information about each token's position in the sequence using sine/cosine functions or learned vectors.

```
Final input = Token Embedding + Positional Encoding
```

#### Multi-Head Self-Attention
The heart of the Transformer. (Covered in detail in Section 8.)

#### Feed-Forward Network (FFN)
After attention, each token's representation passes through a two-layer MLP independently. This is where much of the model's "knowledge" is stored.

```
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

#### Layer Normalization
Applied before or after each sub-layer to stabilize training (prevents exploding/vanishing activations).

#### Residual Connections
Each sub-layer adds its input to its output: `output = LayerNorm(x + SubLayer(x))`. This allows gradients to flow directly through deep networks.

### Scale and Parameters

A model's capability is largely determined by its **parameter count**:

| Model | Parameters | Architecture |
|-------|-----------|-------------|
| BERT-base | 110M | Encoder-only |
| GPT-2 | 1.5B | Decoder-only |
| GPT-3 | 175B | Decoder-only |
| GPT-4 (estimated) | ~1.8T (MoE) | Decoder-only |
| Llama 3 70B | 70B | Decoder-only |
| Mistral 8x7B | ~47B active | Decoder MoE |

---

## 8. Attention Mechanism

### The Core Idea

Attention answers the question: **"When processing this token, which other tokens should I pay attention to?"**

Instead of compressing the entire sequence into a fixed state, attention lets each token directly query all other tokens for relevant information.

### Intuition with an Example

Consider: *"The animal didn't cross the street because **it** was too tired."*

When the model processes "it", attention determines that "it" refers to "animal" not "street":

```
"it" attends strongly to → "animal" (high attention weight)
"it" attends weakly to  → "street" (low attention weight)
```

This is **coreference resolution** happening implicitly through attention.

### Query, Key, Value (QKV) Mechanism

Attention is computed using three learned projections of each token's embedding:

| Component | Analogy | Role |
|-----------|---------|------|
| **Query (Q)** | "What am I looking for?" | The token asking the question |
| **Key (K)** | "What do I offer?" | What each token advertises about itself |
| **Value (V)** | "What information do I provide?" | The actual content retrieved |

**Computation:**

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

1. Compute dot product of Query with all Keys → raw attention scores
2. Scale by √dₖ (prevents vanishing gradients with large dimensions)
3. Apply softmax → attention weights (sum to 1, interpretable as probabilities)
4. Weighted sum of Values using those weights → context-enriched representation

### Multi-Head Attention

Instead of one attention operation, the model runs **H parallel attention heads**, each with different learned Q, K, V projections. The outputs are concatenated and projected.

**Why multiple heads?**
- Each head can specialize in different relationship types
- Head 1 might track syntactic structure (subject-verb agreement)
- Head 2 might track semantic similarity
- Head 3 might track coreference

```python
# Simplified attention score visualization
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)           # Raw scores
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)  # Softmax
    output = weights @ V                        # Weighted values
    return output, weights

# Toy example: 4 tokens, 3-dim embeddings
Q = np.random.randn(4, 3)
K = np.random.randn(4, 3)
V = np.random.randn(4, 3)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print("Attention weights (each row sums to 1):")
print(np.round(attention_weights, 3))
```

### Types of Attention

| Type | Description | Used in |
|------|-------------|---------|
| **Self-attention** | Tokens attend to all other tokens in the same sequence | Encoder, Decoder |
| **Masked self-attention** | Tokens can only attend to previous tokens (causal) | Decoder (during generation) |
| **Cross-attention** | Decoder tokens attend to encoder outputs | Encoder-Decoder models |

---

## 9. Encoder and Decoder Concepts

### The Encoder

**Purpose:** Read and understand the input — create rich contextual representations.

- Uses **bidirectional self-attention**: each token attends to all tokens (left AND right)
- Output: a sequence of context-enriched vectors (one per token)
- Optimized for **understanding tasks**

```
Input:  "The weather is nice today"
         ↓  ↓  ↓  ↓  ↓  ↓
Encoder processes all tokens simultaneously, 
each attending to all others bidirectionally
         ↓  ↓  ↓  ↓  ↓  ↓
Output: Rich contextual embeddings
```

**Use cases:** Classification, NER, semantic search, sentence embeddings
**Example models:** BERT, RoBERTa, DistilBERT

### The Decoder

**Purpose:** Generate output tokens one at a time.

- Uses **masked (causal) self-attention**: token at position i can only attend to positions 0..i (cannot see the future)
- Generates autoregressively: output of step N becomes input of step N+1
- Optimized for **generation tasks**

```
Step 1: [BOS] → "The"
Step 2: [BOS, "The"] → "weather"
Step 3: [BOS, "The", "weather"] → "is"
...continues until [EOS] token
```

**Use cases:** Text generation, language modeling, chat
**Example models:** GPT series, Claude, Llama, Mistral

### Encoder-Decoder (Seq2Seq)

**Purpose:** Encode an input sequence, then decode to produce an output sequence of (potentially) different length.

- Encoder reads source → Decoder generates target
- Cross-attention: decoder attends to encoder's output at each generation step

```
Source: "Hello, how are you?"  → [Encoder] → Context vectors
                                                     ↓
                                              [Decoder]
Target: "Bonjour, comment allez-vous?"  ←  Generated autoregressively
```

**Use cases:** Translation, summarization, document-to-document tasks
**Example models:** T5, BART, mT5

### Summary: Which Architecture for Which Task?

| Architecture | Attention Direction | Strengths | Models |
|-------------|--------------------|-----------|----|
| **Encoder-only** | Bidirectional | Understanding, classification, embeddings | BERT, RoBERTa |
| **Decoder-only** | Causal (left-to-right) | Generation, chat, language modeling | GPT, Claude, Llama |
| **Encoder-Decoder** | Both | Seq2Seq: translation, summarization | T5, BART |

Most modern LLMs (GPT-4, Claude, Llama) are **decoder-only** because:
- Generative capability is the primary goal
- Simpler to scale
- In-context learning and instruction following work naturally

---

## 10. Understanding Large Language Models (LLMs)

### What Makes a Model "Large"?

An LLM is a Transformer-based language model trained at scale:

| Dimension | Small | Medium | Large |
|-----------|-------|--------|-------|
| Parameters | <1B | 1B–10B | 10B–1T+ |
| Training tokens | <100B | 100B–1T | 1T–15T+ |
| Training compute | Single GPU | Multi-GPU | GPU cluster (weeks/months) |
| Context window | 2K–8K | 8K–32K | 128K–1M+ |

### Pre-training: Learning the World from Text

During pre-training, the model is given a simple **self-supervised objective**:

> **Predict the next token** given all previous tokens.

No human labeling needed — the training signal comes from the text itself.

```
Input:  "The capital of France is"
Target: "Paris"

Loss = -log P("Paris" | "The capital of France is")
```

With trillions of such examples across diverse text (web, books, code, scientific papers), the model internalizes:
- Grammar and syntax
- Facts and world knowledge
- Reasoning patterns
- Coding ability
- Mathematical relationships
- Multiple languages

### Emergent Capabilities

As models scale, unexpected abilities **emerge** — capabilities not explicitly trained for and not present in smaller models.

| Capability | Emerges at scale |
|------------|-----------------|
| Multi-step arithmetic | ~7B parameters |
| Chain-of-thought reasoning | ~100B parameters |
| In-context learning (few-shot) | ~10B+ parameters |
| Code generation | ~10B+ parameters |
| Instruction following | After RLHF alignment |

### Instruction Tuning and RLHF

Raw pre-trained models are excellent at text completion but poor at following instructions. Two additional training phases align them:

#### Supervised Fine-Tuning (SFT)
Fine-tune on curated (instruction, response) pairs written by humans:
```
Instruction: "Summarize this article in 3 bullet points."
Response:    "• Point 1\n• Point 2\n• Point 3"
```

#### RLHF — Reinforcement Learning from Human Feedback
1. Generate multiple responses to a prompt
2. Human raters rank them (better → worse)
3. Train a **reward model** on those rankings
4. Use PPO (Proximal Policy Optimization) to maximize the reward model's score

This is what transforms a "text completion engine" into a helpful, safe assistant.

```
Raw GPT:         "Here's how to pick a lock: [continues...]"
After RLHF:      "I can help with lock-related questions. For your own lock..."
```

---

## 11. Context Windows and Token Management

### What is a Context Window?

The **context window** (also called context length or context limit) is the maximum number of tokens an LLM can process in a single call — inputs AND outputs combined.

```
Context Window = Input tokens + Output tokens
```

Everything the model "knows" about the current conversation or task must fit within this window. The model has no memory outside of it.

### Context Windows Across Models

| Model | Context Window |
|-------|---------------|
| GPT-3.5 | 16K tokens |
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Llama 3 70B | 128K tokens |

**128K tokens ≈ 100,000 words ≈ a full novel**

### Why Context Length Matters

| Scenario | Tokens Needed |
|----------|---------------|
| Single Q&A exchange | ~200–500 |
| Multi-turn chat session | ~1,000–5,000 |
| Summarize a 10-page doc | ~8,000–10,000 |
| Analyze an annual report | ~50,000–100,000 |
| Entire codebase review | 200,000+ |

### The Attention Complexity Problem

Self-attention computes relationships between **every pair** of tokens:

```
Complexity = O(n²) where n = sequence length
```

At 128K tokens: 128,000² = 16 billion attention score pairs per layer. This is why:
- Long context inference is expensive (memory and compute)
- Techniques like **sliding window attention**, **sparse attention**, and **KV caching** are used to manage this

### KV Cache

During generation, the model recomputes the same Keys and Values for all previous tokens at every step. **KV caching** stores these, so only the new token's attention needs to be computed each step — dramatically speeding up inference.

### Token Budget Management in Practice

```python
import tiktoken

def estimate_cost(text: str, model: str = "gpt-4o") -> dict:
    enc = tiktoken.encoding_for_model(model)
    token_count = len(enc.encode(text))
    
    # Approximate pricing (check current rates)
    price_per_1k = {"gpt-4o": 0.005, "gpt-4o-mini": 0.00015}
    cost = (token_count / 1000) * price_per_1k.get(model, 0.005)
    
    return {
        "tokens": token_count,
        "estimated_cost_usd": round(cost, 6),
        "approx_words": int(token_count * 0.75)
    }

sample = "Artificial intelligence is transforming every industry." * 100
print(estimate_cost(sample))
```

### Strategies When Input Exceeds Context Limit

| Strategy | When to Use |
|----------|------------|
| **Chunking + summarization** | Summarize chunks; feed summaries |
| **RAG (retrieval)** | Only retrieve the most relevant chunks |
| **Sliding window** | Process overlapping windows; aggregate outputs |
| **Hierarchical summarization** | Summarize sections, then summary of summaries |
| **Use a larger context model** | When precision matters and budget allows |

---

## 12. Prompt-Response Lifecycle

Understanding what happens between hitting "Enter" and seeing a response.

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                               │
│    "Explain transformer attention in simple terms"         │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 2. TOKENIZATION                                             │
│    Text → Token IDs: [9527, 22249, 9554, 287, 3704, 2846] │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 3. CONTEXT ASSEMBLY                                         │
│    System prompt + Conversation history + User message      │
│    = Full context (must fit within context window)          │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 4. EMBEDDING LOOKUP                                         │
│    Token IDs → Dense embedding vectors                      │
│    + Positional encodings added                             │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 5. TRANSFORMER FORWARD PASS                                 │
│    N layers of: Self-Attention → FFN → LayerNorm            │
│    (for GPT-4: ~96 layers, 96 attention heads)              │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 6. NEXT TOKEN PREDICTION                                    │
│    Final hidden state → Linear layer → Logits over vocab    │
│    Logits → Softmax → Probability distribution              │
│    [P("Transformer"=0.42, P("The")=0.18, P("An")=0.12...)] │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 7. SAMPLING / DECODING                                      │
│    Select token from distribution using sampling strategy:  │
│    • Greedy: pick highest probability                        │
│    • Top-k: sample from top k tokens                        │
│    • Top-p (nucleus): sample from top p% of probability     │
│    • Temperature scaling: controls randomness               │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 8. AUTOREGRESSIVE GENERATION                                │
│    Append selected token; repeat steps 5–7                  │
│    Continue until: [EOS] token OR max_tokens reached        │
└──────────────────────────┬──────────────────────────────────┘
                            │
┌──────────────────────────▼──────────────────────────────────┐
│ 9. DETOKENIZATION                                           │
│    Token IDs → Text → Streamed to user                      │
└─────────────────────────────────────────────────────────────┘
```

### Sampling Parameters Explained

| Parameter | Effect | Typical Value |
|-----------|--------|---------------|
| **temperature** | Scales logits before softmax; higher = more random, lower = more deterministic | 0.0–1.0 |
| **top_p** (nucleus) | Only sample from tokens comprising top p% of probability mass | 0.9–0.95 |
| **top_k** | Only sample from top k most likely tokens | 20–50 |
| **max_tokens** | Maximum output length | Task-dependent |
| **stop sequences** | Strings that halt generation | `["\n\n", "###"]` |

```python
# Temperature effect demonstration
from openai import OpenAI
client = OpenAI()

prompt = "Complete this sentence: The best thing about AI is"

for temp in [0.0, 0.5, 1.0, 1.5]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        max_tokens=50
    )
    print(f"Temperature {temp}: {response.choices[0].message.content}")
```

**Expected pattern:**
- `temperature=0.0` → Same deterministic output every run
- `temperature=1.0` → Creative, varied responses
- `temperature=1.5` → Increasingly incoherent/random

---

## 13. Hands-On Exercises

### Exercise 1: Text Processing Exercises

**Goal:** Build intuition for text normalization and the pipeline before modeling.

```python
import re
import string
from collections import Counter

def preprocess_text(text: str) -> dict:
    """Full NLP preprocessing pipeline"""
    
    # 1. Lowercase
    lowercased = text.lower()
    
    # 2. Remove punctuation
    no_punct = text.translate(str.maketrans("", "", string.punctuation))
    
    # 3. Tokenize (simple whitespace split)
    words = no_punct.lower().split()
    
    # 4. Remove stopwords (simplified list)
    stopwords = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
                 "of", "and", "or", "but", "with", "this", "that", "are", "was"}
    filtered = [w for w in words if w not in stopwords]
    
    # 5. Word frequency
    freq = Counter(filtered)
    
    return {
        "original": text,
        "token_count": len(words),
        "unique_words": len(set(words)),
        "top_10_words": freq.most_common(10),
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2)
    }

sample = """
Generative AI is transforming the technology landscape. Large language models 
like GPT-4 and Claude are capable of generating human-like text, writing code, 
and solving complex reasoning problems. These models are trained on vast amounts 
of internet text using transformer architectures with self-attention mechanisms.
"""

result = preprocess_text(sample)
for k, v in result.items():
    print(f"{k}: {v}")
```

---

### Exercise 2: Exploring Tokenization

**Goal:** Understand how different text types tokenize and build cost-estimation intuition.

```python
import tiktoken

enc_gpt4 = tiktoken.encoding_for_model("gpt-4o")

test_cases = {
    "English sentence": "The quick brown fox jumps over the lazy dog.",
    "Technical jargon": "The Transformer's multi-head self-attention mechanism uses QKV projections.",
    "Code snippet": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "Chinese text": "人工智能正在改变世界",
    "Rare/novel word": "antidisestablishmentarianism pneumonoultramicroscopicsilicovolcanoconiosis",
    "Numbers": "The year 2024 has 365 days and 8760 hours.",
    "Emojis": "I love AI! 🤖🧠💡🚀",
    "JSON": '{"name": "Alice", "role": "engineer", "skills": ["python", "ml"]}',
}

print(f"{'Text Type':<30} {'Tokens':>7} {'Characters':>12} {'Chars/Token':>12}")
print("-" * 65)
for label, text in test_cases.items():
    tokens = enc_gpt4.encode(text)
    chars_per_token = round(len(text) / len(tokens), 2)
    print(f"{label:<30} {len(tokens):>7} {len(text):>12} {chars_per_token:>12}")

# Visualize individual tokens
print("\n--- Token breakdown for technical sentence ---")
text = "The Transformer architecture uses multi-head self-attention."
tokens = enc_gpt4.encode(text)
decoded = [enc_gpt4.decode([t]) for t in tokens]
print(f"Tokens ({len(tokens)}): {decoded}")
```

**Observe:**
- English is most token-efficient (~4 chars/token)
- Code tokenizes differently (underscores, brackets are often separate tokens)
- Chinese characters typically use 1–3 tokens per character (less efficient)
- Rare/long words are split into subword pieces

---

### Exercise 3: Working with LLM Interfaces

**Goal:** Explore the key parameters that control LLM behavior and observe their effects.

```python
from openai import OpenAI

client = OpenAI()

def query_llm(
    prompt: str,
    system: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 300
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    return {
        "response": response.choices[0].message.content,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "finish_reason": response.choices[0].finish_reason
    }


# --- Experiment 1: System prompt influence ---
print("=== System Prompt Influence ===")
user_q = "What is machine learning?"

result_default = query_llm(user_q)
result_expert  = query_llm(user_q, system="You are a PhD researcher in ML. Be technical.")
result_simple  = query_llm(user_q, system="You explain concepts to 10-year-olds using analogies.")

print("Default:  ", result_default["response"][:200])
print("Expert:   ", result_expert["response"][:200])
print("Simple:   ", result_simple["response"][:200])


# --- Experiment 2: Multi-turn conversation ---
print("\n=== Multi-turn Conversation ===")
conversation = [
    {"role": "system", "content": "You are an AI tutor teaching NLP concepts."}
]

turns = [
    "What is tokenization?",
    "Can you give me a Python example?",
    "What is the difference between BPE and WordPiece?"
]

for user_turn in turns:
    conversation.append({"role": "user", "content": user_turn})
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation,
        max_tokens=200
    )
    
    assistant_reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": assistant_reply})
    
    print(f"\nUser:      {user_turn}")
    print(f"Assistant: {assistant_reply[:300]}")
    print(f"[Context so far: {response.usage.prompt_tokens} tokens]")


# --- Experiment 3: Finish reason awareness ---
print("\n=== Token Limit Truncation ===")
result_short = query_llm(
    "Write a detailed essay on the history of artificial intelligence.",
    max_tokens=50   # Intentionally too short
)
print(f"Finish reason: {result_short['finish_reason']}")  # 'length' means truncated
print(f"Response: {result_short['response']}")
print("Note: finish_reason='length' means the response was cut off — increase max_tokens")
```

---

## 14. Key Concepts Summary

| Concept | Definition |
|---------|-----------|
| **NLP** | AI field enabling machines to understand and generate human language |
| **Tokenization** | Splitting text into sub-word units (tokens) for model processing |
| **BPE** | Byte-Pair Encoding — merges frequent character pairs into sub-word tokens |
| **Embedding** | Dense vector representation of a token capturing semantic meaning |
| **Semantic similarity** | Closeness of meaning, measured by cosine similarity between embeddings |
| **Transformer** | Neural network architecture using self-attention instead of recurrence |
| **Self-attention** | Mechanism letting each token attend to all other tokens in the sequence |
| **QKV** | Query, Key, Value — three projections used to compute attention weights |
| **Multi-head attention** | Parallel attention heads, each learning different relationship types |
| **Encoder** | Transformer component that reads input bidirectionally (understanding) |
| **Decoder** | Transformer component that generates output token-by-token (generation) |
| **Context window** | Maximum tokens (input + output) a model can process in one call |
| **Autoregressive** | Generating one token at a time, feeding each output back as next input |
| **Temperature** | Sampling parameter: 0 = deterministic, >1 = more random |
| **RLHF** | Reinforcement Learning from Human Feedback — alignment technique |
| **Emergent capability** | Ability that appears at scale not explicitly trained for |
| **KV cache** | Caching Keys/Values from previous tokens to speed up generation |

---

## 15. Module Review Questions

1. What are the four eras of NLP development? What was the key limitation that each new era overcame?
2. Explain why a Transformer can be trained faster than an LSTM on the same dataset.
3. Given the text "unbelievably", how might a BPE tokenizer split it? Why does this matter for unknown words?
4. Two sentences have an embedding cosine similarity of 0.92. What does that tell you about their relationship?
5. What is the difference between an encoder-only and a decoder-only model? Give one real-world use case for each.
6. A user submits a 50-page PDF (approximately 40,000 words) to an LLM with a 32K token context window. What happens, and what are two strategies to handle this?
7. Explain the difference between `temperature=0.0` and `temperature=1.0` in an API call. When would you use each?
8. What is the `finish_reason` field in an API response, and what does `"length"` mean?
9. Why is RLHF necessary after pre-training? What problem does it solve?
10. You're building a semantic search engine over 10,000 product descriptions. Which component of the Transformer stack would you use, and why?

---

## 16. Further Reading & Resources

| Resource | Type | Source |
|----------|------|--------|
| Attention Is All You Need | Original paper | Vaswani et al., 2017 — arXiv:1706.03762 |
| The Illustrated Transformer | Visual explainer | jalammar.github.io |
| The Illustrated BERT | Visual explainer | jalammar.github.io |
| Andrej Karpathy: Let's build GPT | Video walkthrough | YouTube — ~2 hours |
| tiktoken library | Tokenization tool | github.com/openai/tiktoken |
| Tokenizer playground | Interactive tool | platform.openai.com/tokenizer |
| Hugging Face NLP Course | Free course | huggingface.co/learn/nlp-course |
| SBERT: Sentence Transformers | Embeddings library | sbert.net |
| Word2Vec paper | Seminal paper | Mikolov et al., 2013 |

---

## Next Module

**Module 3: Prompt Engineering**

With the mechanics of how LLMs work now clear, Module 3 focuses on how to communicate with them effectively. You'll learn the principles of prompt design, zero-shot and few-shot techniques, role-based prompting, and how to generate structured outputs reliably.
