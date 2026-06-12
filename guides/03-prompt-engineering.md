# Module 3: Prompt Engineering

## Overview

| Attribute | Details |
|-----------|---------|
| Module | 3 of 7 |
| Duration | ~3 Hours |
| Focus | Communicating Effectively with LLMs |

**Learning Objectives**

By the end of this module, you will be able to:
- Identify the structural components of a well-formed prompt
- Apply the core principles of effective prompt design
- Use zero-shot, few-shot, role-based, and instruction-based prompting techniques
- Generate structured outputs (JSON, tables, lists) reliably from LLMs
- Optimize prompts iteratively using systematic techniques
- Evaluate prompt quality and diagnose failure modes

---

## 1. What is Prompt Engineering?

A **prompt** is any text input you send to an LLM to elicit a response. **Prompt engineering** is the discipline of crafting and refining these inputs to reliably produce accurate, relevant, and well-formatted outputs.

The same model, given two different prompts for the same task, can produce outputs that range from useless to exceptional. Prompt engineering is the skill of closing that gap.

### Why It Matters

LLMs do not execute code — they predict continuations of text. This means:
- The model has no inherent understanding of your *intent*, only your *words*
- Ambiguous input → unpredictable output
- Adding context, constraints, and examples directly shapes the probability distribution over outputs
- Better prompts = better results without changing the model or paying for fine-tuning

### Prompt Engineering vs Fine-tuning vs RAG

| Approach | When to Use | Cost | Effort |
|----------|------------|------|--------|
| **Prompt Engineering** | General tasks, rapid iteration | Low (API calls only) | Low–Medium |
| **Fine-tuning** | Consistent style/format, domain vocab | Medium (training run) | High |
| **RAG** | Factual accuracy over private/updated docs | Medium (infra) | Medium |
| **All three combined** | Production-grade enterprise apps | High | High |

Prompt engineering is almost always the **first** thing to try — it is free, fast, and often sufficient.

---

## 2. Prompt Components and Structure

A production-quality prompt typically consists of up to six components. Not all are required for every prompt — simpler tasks need fewer components.

```
┌─────────────────────────────────────────────────────┐
│  SYSTEM PROMPT (optional but recommended)           │
│  Sets persona, rules, and constraints               │
├─────────────────────────────────────────────────────┤
│  CONTEXT / BACKGROUND                               │
│  Relevant information the model needs to know       │
├─────────────────────────────────────────────────────┤
│  EXAMPLES (few-shot)                                │
│  Demonstrations of desired input → output behavior  │
├─────────────────────────────────────────────────────┤
│  INSTRUCTION                                        │
│  The specific task or question                      │
├─────────────────────────────────────────────────────┤
│  INPUT DATA                                         │
│  The content to operate on (document, code, etc.)  │
├─────────────────────────────────────────────────────┤
│  OUTPUT FORMAT SPECIFICATION                        │
│  How the response should be structured              │
└─────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. System Prompt
Sets the model's **persona, role, and behavioral constraints** for the entire interaction. In the OpenAI Chat API, this is the `system` message.

```
You are a senior software engineer specializing in Python.
You give concise, technically accurate answers.
Always include code examples. Never suggest deprecated libraries.
```

#### 2. Context / Background
Provides the **situational knowledge** the model needs but may not have.

```
The application is a financial services platform serving enterprise clients.
It processes 50,000 transactions per day and must meet SOC 2 compliance requirements.
The codebase uses Python 3.11, FastAPI, and PostgreSQL.
```

#### 3. Examples (Few-shot)
Shows the model **what good output looks like** through demonstrations. Covered in depth in Section 6.

#### 4. Instruction
The **explicit task directive** — what you want the model to do.

```
Analyze the following customer review and classify its sentiment.
```

#### 5. Input Data
The **content to act on**, clearly delimited from the instruction.

```
Review: """
The product arrived damaged and customer support was unhelpful.
I would not recommend this to anyone.
"""
```

#### 6. Output Format Specification
Tells the model **exactly how to structure its response**.

```
Respond in JSON with keys: sentiment (positive/negative/neutral),
confidence (0.0-1.0), and key_phrases (list of strings).
```

### Putting It Together

```python
from openai import OpenAI
client = OpenAI()

system = """You are a customer experience analyst.
Classify reviews with precision. Always respond in valid JSON only."""

prompt = """
## Context
These reviews are from an e-commerce platform for electronics.

## Task
Classify the sentiment of the following review.

## Review
"The laptop runs fast and the battery lasts all day. Setup was easy.
 However, the keyboard feels a bit shallow compared to my old one."

## Output Format
{
  "sentiment": "positive | negative | neutral | mixed",
  "confidence": 0.0 to 1.0,
  "key_phrases": ["phrase1", "phrase2"],
  "summary": "one sentence"
}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1   # Low temperature for classification tasks
)
print(response.choices[0].message.content)
```

---

## 3. Principles of Effective Prompt Design

### Principle 1: Be Specific and Explicit

The model cannot read your mind. State the task, constraints, and expected output precisely.

| Vague | Specific |
|-------|----------|
| "Summarize this" | "Summarize this in 3 bullet points, each under 15 words, for a non-technical executive" |
| "Fix the bug" | "Identify and fix the null pointer exception in the `process_order()` function. Explain what caused it." |
| "Write an email" | "Write a professional decline email to a vendor. Tone: polite but firm. Length: under 100 words. No apologies." |

### Principle 2: Provide Context

LLMs have broad world knowledge but no knowledge of your specific situation. Relevant context dramatically improves relevance.

```
❌ "What is the best database for this project?"

✅ "We are building a real-time analytics dashboard for 500 concurrent users.
   Data is append-only (event logs), queries are mostly aggregations over time ranges,
   and the team has strong Python skills but no DBA. What is the best database?"
```

### Principle 3: Use Delimiters to Separate Content

When your prompt contains multiple sections or input data, use **clear delimiters** to prevent the model from confusing instructions with content.

Common delimiters:
- Triple backticks: ` ``` `
- Triple quotes: `"""`
- XML-style tags: `<document>`, `<instructions>`, `<example>`
- Markdown headers: `## Task`, `## Input`, `## Output`
- Dashes: `---`

```
Summarize the text delimited by triple backticks in 2 sentences.

```
The global AI market is projected to grow from $142 billion in 2023 to
over $1.8 trillion by 2030, driven by enterprise adoption, automation,
and advances in foundation models...
```
```

Without delimiters, adversarial inputs can hijack instructions (prompt injection).

### Principle 4: Specify Length, Format, and Tone

```
Write a product description for noise-cancelling headphones.
- Length: 50–75 words
- Format: 2 short paragraphs
- Tone: Professional and aspirational, not hyperbolic
- Audience: Business travelers aged 30–50
- Do NOT use the words "revolutionary" or "game-changing"
```

### Principle 5: Give the Model Room to Think

For complex tasks, instruct the model to reason **before** giving the final answer.

```
❌ "Is this customer eligible for a refund? Customer: bought 35 days ago, item broken."

✅ "Think through the refund eligibility step by step:
   1. State the relevant policy rules
   2. Apply each rule to the customer's situation
   3. Conclude with a yes/no and brief rationale

   Customer: purchased 35 days ago, item arrived broken, standard 30-day policy applies,
   but policy has exception for defective items within 90 days."
```

### Principle 6: Avoid Negations Where Possible

Models sometimes struggle with negations — they are better at following instructions about what TO do.

| Negation (weaker) | Positive instruction (stronger) |
|-------------------|---------------------------------|
| "Don't use bullet points" | "Use numbered lists only" |
| "Don't be too long" | "Respond in exactly 3 sentences" |
| "Don't mention competitors" | "Focus exclusively on our product's strengths" |

### Principle 7: Iterate Systematically

A prompt is not written once — it is refined. Treat prompt development like software development:
1. Write a baseline version
2. Test against diverse inputs
3. Identify failure modes
4. Add constraints or examples to fix them
5. Regression-test that previous cases still pass
6. Document the final prompt with rationale for key decisions

---

## 4. Zero-Shot Prompting

**Zero-shot prompting** means asking the model to perform a task with **no examples** — relying entirely on the model's pre-trained knowledge.

### When It Works

- Well-known task types (translation, summarization, classification)
- Tasks where the desired format is implied by the task name
- Exploratory use where you want the model's "best guess"

### Anatomy of a Zero-Shot Prompt

```
[Task description] + [Input data] + [Output specification (optional)]
```

### Examples

**Classification:**
```
Classify the sentiment of this review as positive, negative, or neutral.

Review: "Delivery was fast but the packaging was completely destroyed."

Answer with a single word.
```

**Translation:**
```
Translate the following English text to formal Spanish.

Text: "We appreciate your patience and will resolve this issue by Friday."
```

**Extraction:**
```
Extract all dates mentioned in the following contract clause.
Return them as a comma-separated list in YYYY-MM-DD format.

Clause: "The agreement commences on January 15, 2025, with a review
on the first of March and final delivery by December 31st, 2025."
```

**Summarization:**
```
Summarize the key findings from this research abstract in one sentence,
written for a business audience unfamiliar with technical jargon.

Abstract: [paste text here]
```

### Limitations of Zero-Shot

- Fails on tasks requiring **domain-specific output format**
- Inconsistent on **edge cases** without examples to anchor behavior
- Struggles with **novel or unusual tasks** the model wasn't exposed to at scale

---

## 5. Few-Shot Prompting

**Few-shot prompting** provides the model with **2–10 examples** of the desired input-output behavior before the actual task. This conditions the model's output distribution toward your expected format and style.

### Why It Works

The model doesn't "learn" from these examples (no weight updates). Instead, in-context examples shift the probability distribution of outputs — the model pattern-matches the structure and style of your examples.

### Structure of a Few-Shot Prompt

```
[Task description / instruction]

Example 1:
Input: [input_1]
Output: [output_1]

Example 2:
Input: [input_2]
Output: [output_2]

Example 3:
Input: [input_3]
Output: [output_3]

Now complete this:
Input: [actual_input]
Output:
```

### Example: Sentiment Classification with Confidence

```python
from openai import OpenAI
client = OpenAI()

few_shot_prompt = """
Classify customer support tickets by urgency (low/medium/high) and category.

---
Ticket: "Hi, could you update my billing address when you get a chance?"
Urgency: low
Category: billing
Reason: Non-urgent account update request

---
Ticket: "My account has been hacked and someone is making unauthorized purchases RIGHT NOW!"
Urgency: high
Category: security
Reason: Active security incident with financial impact

---
Ticket: "The mobile app crashes when I try to view my transaction history."
Urgency: medium
Category: technical
Reason: Functional issue affecting usability but not a security/financial emergency

---
Ticket: "I was charged twice for my subscription this month and need a refund immediately."
Urgency:"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": few_shot_prompt}],
    temperature=0.0,
    max_tokens=100
)
print(response.choices[0].message.content)
```

### Guidelines for Effective Few-Shot Examples

| Guideline | Rationale |
|-----------|-----------|
| **Cover diverse cases** | Don't just show easy examples — include edge cases |
| **Consistent format** | Every example must follow the exact same input/output structure |
| **Label quality matters** | Wrong examples teach the wrong behavior |
| **3–5 examples** is usually sufficient | More examples consume context and have diminishing returns |
| **Order can matter** | Most recent examples have more influence (recency bias) |
| **Match distribution** | Examples should reflect the real distribution of inputs you'll encounter |

### Zero-Shot vs Few-Shot Comparison

```python
PROMPT_ZERO_SHOT = """
Extract the company name, role, and years of experience from this resume line.
Respond as JSON.

Resume line: "Led a team of 12 engineers at Stripe for 4 years as Principal Engineer"
"""

PROMPT_FEW_SHOT = """
Extract the company name, role, and years of experience from resume lines.
Respond as JSON.

Resume line: "Senior Data Scientist at Netflix for 3 years"
{"company": "Netflix", "role": "Senior Data Scientist", "years": 3}

Resume line: "Worked as a DevOps Engineer at Shopify from 2019 to 2022"
{"company": "Shopify", "role": "DevOps Engineer", "years": 3}

Resume line: "Led a team of 12 engineers at Stripe for 4 years as Principal Engineer"
"""
```

Few-shot is especially valuable when:
- Output format must be precise (JSON keys, date format, enum values)
- The task involves domain-specific classification labels
- Zero-shot produces inconsistent results across runs

---

## 6. Role-Based Prompting

**Role-based prompting** assigns the model a specific **persona or expert identity**, which shapes its tone, vocabulary, depth, and perspective.

### How Roles Influence Output

When you tell the model "You are a [role]", it:
- Adopts appropriate domain vocabulary
- Adjusts depth of explanation to the role's expertise
- Applies the role's conventions and norms
- Filters output through the role's perspective and priorities

### Role Prompt Patterns

#### Expert Persona
```
You are a board-certified cardiologist with 20 years of clinical experience.
Explain the risks of untreated hypertension in terms a patient can understand.
Avoid jargon. Emphasize when to seek urgent care.
```

#### Audience-Calibrated Persona
```
You are a teacher explaining programming concepts to 10-year-olds.
Use simple analogies from everyday life. Avoid all technical terminology.
Check for understanding by asking one question at the end.
```

#### Adversarial/Critical Persona
```
You are a strict code reviewer with a focus on security vulnerabilities.
Review the following Python code. Be skeptical. Identify every potential
security issue, even minor ones. Do not compliment the code.
```

#### Structured Reasoning Persona
```
You are a management consultant using the MECE (Mutually Exclusive, Collectively
Exhaustive) framework. Analyze the following business problem and structure your
response as a MECE issue tree.
```

### Role + Audience Combination

Roles become most powerful when you also specify the **target audience**:

```python
base_question = "Explain how a neural network learns."

roles = {
    "researcher": "You are an ML researcher. Be technically precise. Use math where helpful.",
    "executive": "You are a business strategist. Focus on implications, not mechanics. Avoid math.",
    "teacher": "You are a high school science teacher. Use memorable analogies. No assumed background.",
    "engineer": "You are a senior software engineer. Focus on implementation. Include pseudocode."
}

for role_name, system_prompt in roles.items():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_question}
        ],
        max_tokens=150
    )
    print(f"\n=== {role_name.upper()} ===")
    print(response.choices[0].message.content)
```

### Common Role Templates

| Role | Use Case | Key Constraints to Add |
|------|----------|------------------------|
| Senior [language] engineer | Code review, generation | "Follow PEP 8", "No deprecated APIs" |
| Technical writer | Documentation | "Audience: developers", "Include code examples" |
| Data analyst | Data interpretation | "Be quantitative", "Highlight outliers" |
| Legal reviewer | Contract analysis | "Flag risks", "Do not give legal advice" |
| Devil's advocate | Strategy review | "Challenge every assumption" |
| Socratic tutor | Learning/teaching | "Ask questions, don't give answers directly" |

### Role Limitations and Risks

- Roles **do not grant real expertise** — the model's underlying knowledge is unchanged
- For high-stakes domains (medical, legal, financial), always add: **"This is for informational purposes only. Recommend the user consult a licensed professional."**
- Roles can create **sycophantic drift** in long conversations — periodically re-anchor with the system prompt

---

## 7. Instruction-Based Prompting

**Instruction-based prompting** focuses on giving the model **explicit, structured directives** — breaking down the task into steps, rules, and constraints rather than relying on the model to infer intent.

### Imperative Instructions

Use clear imperative verbs at the start of instructions:

| Verb | Use |
|------|-----|
| **Classify** | Assign categories |
| **Extract** | Pull out specific data |
| **Summarize** | Condense |
| **Translate** | Convert languages |
| **Generate** | Create new content |
| **Compare** | Identify similarities and differences |
| **Rewrite** | Modify existing text |
| **Evaluate** | Assess against criteria |
| **List** | Enumerate items |
| **Explain** | Provide reasoning or description |

### Step-by-Step Instructions

For multi-stage tasks, enumerate each step explicitly:

```
Analyze the following customer complaint and respond as follows:

Step 1: Identify the core issue in one sentence.
Step 2: Determine the sentiment (frustrated / disappointed / angry / confused).
Step 3: List the specific products or services mentioned.
Step 4: Draft a professional empathetic response (under 80 words).
Step 5: Rate the urgency from 1 (low) to 5 (high) with a one-line justification.

Complaint: """
I ordered the premium plan three weeks ago and still can't access half the features.
I've contacted support twice and each time I'm told it will be fixed "soon."
This is completely unacceptable for what I'm paying.
"""
```

### Constraint Instructions

Constraints shape what the model should and should not do:

```
Write a product tagline for a project management tool.

Constraints:
- Maximum 8 words
- Must include a verb
- Do NOT use the words: "easy", "simple", "powerful", "smart"
- Target audience: software development teams
- Tone: confident and direct
```

### Chain-of-Thought (CoT) Instruction

Instructing the model to **show its reasoning** before answering improves accuracy on reasoning-heavy tasks:

```python
cot_prompt = """
A company has 3 products. Product A has 40% market share, 
Product B has 35%, and Product C has the rest. 
Revenue is $2.4M total. Product A's revenue per market share point is 1.5x 
that of Products B and C.

Think through this step by step, then state the revenue for each product.
"""

# vs the weaker version:
direct_prompt = """
Same math problem. What is the revenue for each product?
"""
```

**Why CoT works:** The model "writes out" intermediate reasoning steps, which become context for the final answer — reducing the chance of arithmetic or logical errors.

### Instruction Chaining (Multi-Prompt Pipelines)

For complex tasks, break the work into a **pipeline of prompts** rather than one monolithic prompt:

```
Stage 1: Extract key facts from the document
Stage 2: Classify each fact by relevance
Stage 3: Synthesize a structured summary from the relevant facts
Stage 4: Generate action items based on the summary
```

```python
def run_pipeline(document: str) -> dict:
    
    # Stage 1: Extract
    facts_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"List every factual claim in this document as bullet points:\n\n{document}"}]
    )
    facts = facts_response.choices[0].message.content

    # Stage 2: Classify relevance
    relevance_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"From these facts, mark each as HIGH or LOW business relevance:\n\n{facts}"}]
    )
    classified = relevance_response.choices[0].message.content

    # Stage 3: Summarize
    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Write a 3-sentence executive summary from the HIGH relevance facts:\n\n{classified}"}]
    )
    summary = summary_response.choices[0].message.content

    return {"facts": facts, "classified": classified, "summary": summary}
```

---

## 8. Structured Output Generation

One of the most valuable prompt engineering skills is getting LLMs to produce **machine-readable, structured output** — JSON, CSV, XML, Markdown tables — reliably.

### Why Structure Matters

In applications, you need to:
- Parse the model's output programmatically
- Insert values into databases or other systems
- Pass outputs from one prompt to the next
- Display formatted data in UI components

Unstructured prose is hard to parse reliably. Structured outputs bridge the LLM and your application code.

### Technique 1: Explicit JSON Instructions

```python
import json
from openai import OpenAI

client = OpenAI()

def extract_person_info(bio: str) -> dict:
    prompt = f"""
Extract structured information from this biography.
Return ONLY valid JSON. Do not include any explanation or markdown.

Required fields:
- name (string)
- age (integer or null if not mentioned)
- occupation (string)
- skills (array of strings)
- location (string or null)

Biography:
{bio}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    raw = response.choices[0].message.content.strip()
    
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw)

bio = """
Maria Chen is a 34-year-old machine learning engineer based in San Francisco.
She has deep expertise in PyTorch, distributed training, and MLOps.
She previously worked on recommendation systems and is now focused on LLM fine-tuning.
"""

result = extract_person_info(bio)
print(json.dumps(result, indent=2))
```

### Technique 2: OpenAI Structured Outputs (JSON Schema Mode)

The most reliable approach — the API guarantees valid JSON conforming to your schema:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class SentimentAnalysis(BaseModel):
    sentiment: str          # "positive" | "negative" | "neutral" | "mixed"
    confidence: float       # 0.0 to 1.0
    key_phrases: list[str]
    summary: str

def analyze_sentiment(text: str) -> SentimentAnalysis:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze the sentiment of the given text."},
            {"role": "user", "content": text}
        ],
        response_format=SentimentAnalysis,
        temperature=0.0
    )
    return response.choices[0].message.parsed

review = "The conference was well-organized but the keynote speaker ran 30 minutes over."
result = analyze_sentiment(review)
print(f"Sentiment: {result.sentiment} ({result.confidence:.0%} confidence)")
print(f"Key phrases: {result.key_phrases}")
print(f"Summary: {result.summary}")
```

### Technique 3: Markdown Tables

```
Analyze these 5 programming languages and compare them.

Return your analysis as a Markdown table with columns:
Language | Primary Use Case | Learning Curve | Performance | Best For

Include exactly 5 rows, one per language: Python, JavaScript, Go, Rust, Java.
```

### Technique 4: Consistent List Formats

```
List the top 5 risks of deploying an LLM in a healthcare application.

Format each risk as:
**Risk N: [Risk Name]**
- Likelihood: High / Medium / Low
- Impact: High / Medium / Low
- Mitigation: [one sentence]
```

### Handling JSON Extraction Failures

Even with instructions, models occasionally produce malformed JSON. Defensive parsing:

```python
import json
import re

def safe_parse_json(raw: str) -> dict | None:
    # Attempt 1: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Attempt 2: extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Attempt 3: find first { ... } block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None  # All attempts failed — trigger retry logic
```

### Structured Output Best Practices

| Practice | Why |
|----------|-----|
| Set `temperature=0.0` for structured outputs | Reduces randomness; more deterministic structure |
| Specify field names, types, and allowed values explicitly | Model has less to infer |
| Use `response_format={"type": "json_object"}` (OpenAI) | Prevents prose wrapping |
| Add "Do not include any explanation or markdown" | Prevents prose leaking into JSON |
| Validate and retry on parse failure | Models occasionally produce invalid JSON |
| Use Pydantic + `.parse()` for critical production code | Guarantees schema conformance |

---

## 9. Prompt Optimization Techniques

Getting a prompt to work once is not the same as making it work reliably across diverse inputs. Optimization makes prompts robust.

### Technique 1: Temperature Tuning

| Task Type | Recommended Temperature |
|-----------|------------------------|
| Classification, extraction, structured output | 0.0 |
| Factual Q&A, code generation | 0.0–0.3 |
| Summarization, analysis | 0.3–0.5 |
| Creative writing, brainstorming | 0.7–1.0 |
| Highly creative/experimental | 1.0–1.5 |

### Technique 2: Prompt Decomposition

Break a complex single prompt into multiple targeted prompts:

```
❌ One complex prompt:
"Read this 50-page report, extract all financial figures, compare them to
last year's numbers which I'll also paste, identify anomalies, and write
an executive summary with recommendations."

✅ Decomposed pipeline:
Prompt 1: "Extract all financial figures from this year's report as a JSON list."
Prompt 2: "Extract all financial figures from last year's report as a JSON list."
Prompt 3: "Compare these two JSON datasets. Flag any year-over-year changes >15%."
Prompt 4: "Write a 3-paragraph executive summary based on these anomalies."
```

### Technique 3: Prompt Chaining with Validation

```python
def validated_extraction(text: str, max_retries: int = 3) -> dict:
    prompt = f"""
Extract the meeting details from this text as JSON.
Required fields: date, time, attendees (list), location, agenda_items (list).

Text: {text}

Return ONLY valid JSON.
"""
    for attempt in range(max_retries):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        result = safe_parse_json(response.choices[0].message.content)
        
        if result and all(k in result for k in ["date", "time", "attendees", "location"]):
            return result
        
        # Repair prompt on retry
        prompt += "\n\nIMPORTANT: Your previous response was not valid JSON. Return ONLY the JSON object, nothing else."
    
    raise ValueError(f"Failed to extract valid JSON after {max_retries} attempts")
```

### Technique 4: Self-Consistency

For reasoning-heavy tasks, generate **multiple responses** and pick the most common answer:

```python
def self_consistent_answer(question: str, n: int = 5) -> str:
    responses = []
    
    for _ in range(n):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"{question}\n\nThink step by step, then give your final answer on the last line starting with 'Answer:'"}
            ],
            temperature=0.7   # Deliberately varied
        )
        text = response.choices[0].message.content
        # Extract the final answer line
        for line in reversed(text.split("\n")):
            if line.startswith("Answer:"):
                responses.append(line.replace("Answer:", "").strip())
                break
    
    # Return majority vote
    from collections import Counter
    return Counter(responses).most_common(1)[0][0]
```

### Technique 5: Negative Examples

Show the model what **bad output looks like** alongside good output:

```
Classify support tickets as urgent or non-urgent.

GOOD EXAMPLE:
Ticket: "App crashes on login every time"
Classification: urgent
Reason: Core functionality blocked

BAD EXAMPLE (do not do this):
Ticket: "App crashes on login every time"
Classification: non-urgent
Reason: Crashes happen sometimes (WRONG — a login crash is always urgent)

Now classify:
Ticket: "I can't export my reports to PDF"
Classification:
```

### Technique 6: Context Window Optimization

Reduce token consumption without sacrificing quality:

| Technique | Example |
|-----------|---------|
| Abbreviate labels | "pos/neg/neu" instead of "positive/negative/neutral" |
| Compact JSON keys | `{"s": "pos", "c": 0.9}` instead of `{"sentiment": "positive", "confidence": 0.9}` |
| Strip boilerplate from documents | Remove headers, footers, page numbers before injecting |
| Truncate to relevant sections | Use a fast/cheap model to identify relevant sections first |
| Use system prompt efficiently | Put static instructions in `system`, not repeated in every user message |

---

## 10. Prompt Evaluation and Refinement

A prompt that works on 3 test cases is not necessarily production-ready. Systematic evaluation prevents surprises.

### Evaluation Dimensions

| Dimension | Question | Measurement |
|-----------|----------|-------------|
| **Accuracy** | Is the output factually correct? | % correct on labeled test set |
| **Format compliance** | Does output match the required structure? | % valid JSON / table / list |
| **Consistency** | Does the same input always produce equivalent output? | Variance across N runs |
| **Robustness** | Does it handle edge cases gracefully? | Pass rate on adversarial inputs |
| **Latency** | How fast does it respond? | P50/P95 response time |
| **Token efficiency** | How many tokens consumed per call? | Avg tokens per request |

### Building a Test Suite

```python
import json
from dataclasses import dataclass
from typing import Callable

@dataclass
class TestCase:
    input: str
    expected_sentiment: str
    description: str

TEST_SUITE = [
    TestCase("This product is amazing!", "positive", "clear positive"),
    TestCase("Worst purchase I ever made.", "negative", "clear negative"),
    TestCase("It works as described.", "neutral", "neutral factual"),
    TestCase("Love the design, hate the price.", "mixed", "genuinely mixed"),
    TestCase("ok", "neutral", "minimal input"),
    TestCase("!!!!!!", "neutral", "no semantic content"),
    TestCase(
        "The product is not bad and I don't dislike it.",
        "neutral",
        "double negation"
    ),
]

def evaluate_prompt(prompt_fn: Callable[[str], str]) -> dict:
    results = []
    
    for tc in TEST_SUITE:
        try:
            output = prompt_fn(tc.input)
            data = json.loads(output)
            predicted = data.get("sentiment", "").lower()
            passed = predicted == tc.expected_sentiment
        except Exception as e:
            predicted = f"ERROR: {e}"
            passed = False
        
        results.append({
            "description": tc.description,
            "input": tc.input,
            "expected": tc.expected_sentiment,
            "predicted": predicted,
            "passed": passed
        })
    
    pass_rate = sum(r["passed"] for r in results) / len(results)
    print(f"\nPass rate: {pass_rate:.0%} ({sum(r['passed'] for r in results)}/{len(results)})")
    
    for r in results:
        status = "✓" if r["passed"] else "✗"
        print(f"{status} [{r['description']}] Expected: {r['expected']}, Got: {r['predicted']}")
    
    return results
```

### Iterative Refinement Workflow

```
Version 1: Basic zero-shot prompt
    ↓ Test → 60% pass rate
    ↓ Failure analysis: model says "neutral" for mixed reviews

Version 2: Add "mixed" as a label option + definition
    ↓ Test → 75% pass rate
    ↓ Failure analysis: double negations confuse the model

Version 3: Add few-shot example showing double negation handling
    ↓ Test → 90% pass rate
    ↓ Failure analysis: minimal inputs ("ok", "sure") vary across runs

Version 4: Add temperature=0.0, add tiebreaker rule for ambiguous inputs
    ↓ Test → 96% pass rate ← Production threshold
```

### Prompt Versioning

Treat prompts like code — version them:

```python
PROMPTS = {
    "sentiment_v1": "Classify: {text}",
    "sentiment_v2": "Classify as positive/negative/neutral: {text}. Return JSON.",
    "sentiment_v3": """...(current production version)...""",
}

ACTIVE_PROMPT = "sentiment_v3"
```

### Common Failure Modes and Fixes

| Failure Mode | Symptom | Fix |
|-------------|---------|-----|
| **Format drift** | Output ignores structure instructions | Add few-shot examples; use JSON mode |
| **Hallucination** | Model invents facts not in input | Add "Only use information from the provided text" |
| **Over-generalization** | Model adds unsolicited commentary | "Do not add explanations or caveats" |
| **Truncation** | Response cuts off mid-sentence | Increase `max_tokens`; check `finish_reason` |
| **Instruction forgetting** | Long conversations lose early rules | Repeat key rules periodically; use system prompt |
| **Inconsistency** | Same input, different outputs | Lower temperature; add explicit tie-breaking rules |
| **Sycophancy** | Model always agrees with the user | "Do not change your answer based on user pushback unless they provide new evidence" |
| **Literal interpretation** | Model follows instructions too rigidly | Add "Use your judgment for edge cases not covered" |

---

## 11. Hands-On Exercises

### Exercise 1: Prompt Engineering Exercises

**Goal:** Experience how small prompt changes cause large output differences.

```python
from openai import OpenAI
client = OpenAI()

def prompt(user_text: str, system_text: str = "You are a helpful assistant.",
           temperature: float = 0.7, max_tokens: int = 200) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

ARTICLE = """
Quantum computing uses quantum mechanical phenomena to perform calculations.
Unlike classical bits (0 or 1), qubits can exist in superposition — simultaneously
0 and 1. This enables quantum computers to evaluate many solutions in parallel.
Current quantum computers are error-prone and operate only at near absolute zero
temperatures. IBM, Google, and IonQ lead commercial development. Google claimed
"quantum supremacy" in 2019 by solving a problem in 200 seconds that would take
a classical supercomputer 10,000 years.
"""

# Experiment A: Different length constraints
for length in ["1 sentence", "3 bullet points", "a 50-word paragraph", "a tweet (280 chars)"]:
    print(f"\n--- Summarize as {length} ---")
    print(prompt(f"Summarize this article as {length}:\n\n{ARTICLE}"))

# Experiment B: Different audience calibration
for audience in ["a 10-year-old", "a business executive", "a PhD physicist"]:
    print(f"\n--- For {audience} ---")
    print(prompt(f"Explain the main point of this article to {audience}:\n\n{ARTICLE}"))

# Experiment C: Different task types on same content
tasks = {
    "FAQ": "Generate 3 FAQ questions and answers based on this article.",
    "Social post": "Write a LinkedIn post sharing the key insight from this article.",
    "Quiz": "Write a 3-question multiple choice quiz based on this article.",
    "Critique": "Identify what important context or caveats this article is missing."
}
for task_name, instruction in tasks.items():
    print(f"\n--- Task: {task_name} ---")
    print(prompt(f"{instruction}\n\nArticle:\n{ARTICLE}"))
```

---

### Exercise 2: Prompt Optimization Techniques

**Goal:** Take a weak baseline prompt and iteratively improve it until it passes a test suite.

```python
import json

# --- Target task: extract action items from meeting notes ---

MEETING_NOTES = """
Weekly sync - June 10, 2025

Sarah will update the API documentation by end of week.
John needs to review the security audit report before the board meeting on June 20.
The team agreed to migrate to Python 3.12 next sprint.
Alex should follow up with the vendor about the delayed hardware shipment.
Budget approval for the new ML cluster is pending - finance team to confirm by June 15.
"""

# Version 1: Vague baseline
v1 = "What are the action items from these meeting notes?\n\n" + MEETING_NOTES

# Version 2: Add structure
v2 = f"""
Extract all action items from the following meeting notes.
For each action item, identify: who is responsible, what the task is, and the deadline.

Meeting notes:
{MEETING_NOTES}
"""

# Version 3: Add output format
v3 = f"""
Extract all action items from the meeting notes below.

Return a JSON array where each item has:
- "owner": person responsible (string)
- "task": description of the action (string)
- "deadline": due date or "unspecified" if not mentioned (string)

Return ONLY the JSON array. No explanation.

Meeting notes:
{MEETING_NOTES}
"""

# Version 4: Add system prompt + validation
system_v4 = "You are a meeting summarizer. Extract action items precisely. Return only valid JSON."

v4 = f"""
Extract all action items from these meeting notes.

Each action item must include:
- "owner": exact name of the responsible person
- "task": clear description of what must be done
- "deadline": specific date in YYYY-MM-DD format, or "unspecified"

Rules:
- Include ONLY items with a clear owner and task
- "The team" is a valid owner
- Do NOT infer deadlines not explicitly stated

Return a JSON array only.

Meeting notes:
{MEETING_NOTES}
"""

for version, (sys, usr) in enumerate([
    ("You are a helpful assistant.", v1),
    ("You are a helpful assistant.", v2),
    ("You are a helpful assistant.", v3),
    (system_v4, v4)
], start=1):
    print(f"\n{'='*50}")
    print(f"VERSION {version}")
    print('='*50)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": usr}
        ],
        temperature=0.0
    )
    output = response.choices[0].message.content
    print(output)
    try:
        parsed = json.loads(output)
        print(f"[Valid JSON ✓] — {len(parsed)} action items extracted")
    except:
        print("[Invalid JSON ✗]")
```

---

### Exercise 3: Generating Structured Outputs

**Goal:** Build a reusable structured extraction function backed by Pydantic validation.

```python
from pydantic import BaseModel, Field
from openai import OpenAI
import json

client = OpenAI()

# --- Define output schemas ---

class JobPosting(BaseModel):
    job_title: str
    company: str
    location: str
    employment_type: str = Field(description="full-time, part-time, contract, or remote")
    required_skills: list[str]
    experience_years_min: int | None
    salary_range: str | None = Field(description="e.g. '$120k-$150k' or null if not mentioned")
    key_responsibilities: list[str] = Field(max_length=5)

class ProductReview(BaseModel):
    product_name: str
    overall_rating: float = Field(ge=1.0, le=5.0)
    pros: list[str]
    cons: list[str]
    recommended: bool
    target_customer: str = Field(description="Who would benefit most from this product")

# --- Generic extraction function ---

def extract_structured(text: str, schema: type[BaseModel], context: str = "") -> BaseModel:
    system = f"You are a precise data extractor. {context} Return data matching the provided schema exactly."
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Extract structured data from this text:\n\n{text}"}
        ],
        response_format=schema,
        temperature=0.0
    )
    return response.choices[0].message.parsed

# --- Test with job posting ---
job_text = """
Senior ML Engineer – Anthropic (San Francisco, CA / Remote)
Full-time | $180,000 – $240,000

We're looking for an experienced ML Engineer to join our alignment research team.
You'll design training pipelines for large-scale language models, implement RLHF workflows,
and collaborate closely with researchers.

Requirements:
- 5+ years ML engineering experience
- Proficiency in PyTorch, JAX, and distributed training
- Experience with transformers and fine-tuning
- Strong Python skills
- Prior LLM work a plus
"""

job = extract_structured(job_text, JobPosting, "Extract job posting details.")
print("\n=== Job Posting ===")
print(json.dumps(job.model_dump(), indent=2))

# --- Test with product review ---
review_text = """
I've been using the Sony WH-1000XM5 headphones for 3 months now.
The noise cancellation is hands-down the best I've ever used — complete silence on flights.
Battery life is excellent at 30 hours. Sound quality is warm and detailed.

On the downside, they're quite expensive at $350, and the ear cups get warm after 2 hours.
The touch controls take some getting used to.

Overall, if you travel frequently for work and need focus, these are worth every penny.
Rating: 4.5/5
"""

review = extract_structured(review_text, ProductReview, "Extract product review details.")
print("\n=== Product Review ===")
print(json.dumps(review.model_dump(), indent=2))
```

---

## 12. Advanced Prompt Patterns Reference

### Pattern: Persona + Chain-of-Thought + Structured Output

```
You are a risk analyst at a financial institution.

Evaluate the following loan application by thinking through each risk factor step by step,
then produce a structured risk assessment.

Step 1: Assess credit risk (payment history, debt-to-income ratio)
Step 2: Assess collateral risk (asset value vs loan amount)
Step 3: Assess market risk (industry stability, employment outlook)
Step 4: Assign an overall risk rating: LOW / MEDIUM / HIGH / DECLINE

Return your final assessment as JSON:
{
  "credit_risk": "low|medium|high",
  "collateral_risk": "low|medium|high",
  "market_risk": "low|medium|high",
  "overall_rating": "LOW|MEDIUM|HIGH|DECLINE",
  "key_concerns": ["concern1", "concern2"],
  "recommendation": "one sentence"
}

Application: [paste here]
```

### Pattern: ReAct (Reason + Act)

Used in agent frameworks — model interleaves thinking and action:
```
You have access to tools: search(query), calculate(expression), lookup(id).

Thought: I need to find X
Action: search("X")
Observation: [result]
Thought: Based on this, I should...
Action: calculate("...")
Observation: [result]
Final Answer: ...
```

### Pattern: Reflexion / Self-Critique

```
Task: [describe task]

First, provide your answer.
Then, critically evaluate your own answer:
- What might be wrong or missing?
- What assumptions did you make?
- What would you change?

Finally, provide an improved version of your answer.
```

---

## 13. Key Concepts Summary

| Concept | Definition |
|---------|-----------|
| **Prompt** | Text input to an LLM that elicits a response |
| **Prompt Engineering** | Crafting and iterating prompts to reliably produce quality outputs |
| **System prompt** | Persistent instruction that sets persona and rules for the conversation |
| **Zero-shot** | Prompting with no examples; relies on pre-trained knowledge |
| **Few-shot** | Providing 2–10 input-output examples to demonstrate desired behavior |
| **Role-based prompting** | Assigning an expert persona to shape tone and depth |
| **Chain-of-Thought (CoT)** | Instructing the model to reason step by step before answering |
| **Structured output** | Getting the model to produce parseable JSON, CSV, or formatted text |
| **Temperature** | Controls output randomness; 0=deterministic, 1+=creative |
| **Prompt decomposition** | Breaking complex tasks into a pipeline of simpler prompts |
| **Self-consistency** | Generating multiple responses and taking majority vote |
| **Prompt injection** | Adversarial input that hijacks prompt instructions |
| **Sycophancy** | Model agrees with user pushback even when it was originally correct |
| **Hallucination** | Model generates confident but factually incorrect information |
| **Token efficiency** | Minimizing token usage while maintaining output quality |

---

## 14. Module Review Questions

1. Name the six components of a well-structured prompt. Which are always required, and which are optional?
2. You ask a model "Don't make the response too long." Why is this a weak instruction, and how would you rewrite it?
3. What is the key difference between zero-shot and few-shot prompting? When would you choose one over the other?
4. A sentiment classifier works on 90% of your test cases but fails on double-negation inputs ("not bad", "can't complain"). What is the most targeted fix?
5. You need an LLM to always return valid JSON. List three techniques you would use to maximize format reliability.
6. A colleague uses `temperature=1.0` for a JSON extraction task and complains the output is sometimes invalid. What do you advise?
7. What is Chain-of-Thought prompting, and for what types of tasks does it most improve accuracy?
8. Describe the self-consistency technique. What is the tradeoff of using it?
9. What is prompt injection, and how do delimiters help prevent it?
10. You have a production prompt that achieves 94% accuracy on your test suite. Describe three dimensions you would still monitor in production.

---

## 15. Further Reading & Resources

| Resource | Type | Source |
|----------|------|--------|
| Prompt Engineering Guide | Comprehensive guide | promptingguide.ai |
| OpenAI Prompt Engineering | Official docs | platform.openai.com/docs/guides/prompt-engineering |
| Anthropic Prompt Library | Curated examples | docs.anthropic.com/prompt-library |
| Chain-of-Thought Prompting Paper | Research paper | Wei et al., 2022 — arXiv:2201.11903 |
| Self-Consistency Paper | Research paper | Wang et al., 2022 — arXiv:2203.11171 |
| ReAct Paper | Agentic reasoning | Yao et al., 2022 — arXiv:2210.03629 |
| OpenAI Structured Outputs | API docs | platform.openai.com/docs/guides/structured-outputs |
| Pydantic Documentation | Validation library | docs.pydantic.dev |
| LMSYS Chatbot Arena | Model comparison | lmarena.ai |
| LearnPrompting.org | Free course | learnprompting.org |

---

## Next Module

**Module 4: OpenAI APIs & AI Application Integration**

With prompt engineering mastered, Module 4 moves from interactive prompting to programmatic integration. You'll work directly with the OpenAI API in Python — handling authentication, model selection, request parameters, response parsing, and error handling — and build a complete API-driven application.
