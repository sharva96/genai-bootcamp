"""
Exercise 1: OpenAI API Basics
==============================
Run: uv run python exercises/01_api_basics.py

Goals:
- Initialize the OpenAI client
- Make your first chat completion call
- Inspect the full response object
- Understand token usage
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()  # reads OPENAI_API_KEY from environment


# ── Part A: Your First API Call ────────────────────────────────────────────────

def first_call() -> None:
    print("=" * 60)
    print("PART A: First API Call")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user",   "content": "What is an API? Answer in exactly one sentence."}
        ],
        temperature=0.0,
        max_tokens=100,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Finish reason: {response.choices[0].finish_reason}")
    print(f"Model used: {response.model}")
    print(f"Tokens — prompt: {response.usage.prompt_tokens}, "
          f"completion: {response.usage.completion_tokens}, "
          f"total: {response.usage.total_tokens}")


# ── Part B: Inspect the Full Response Object ───────────────────────────────────

def inspect_response() -> None:
    print("\n" + "=" * 60)
    print("PART B: Full Response Object")
    print("=" * 60)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello in 3 words."}],
        temperature=0.5,
        max_tokens=20,
    )

    print(f"ID:             {response.id}")
    print(f"Model:          {response.model}")
    print(f"Created (unix): {response.created}")
    print(f"Content:        {response.choices[0].message.content}")
    print(f"Role:           {response.choices[0].message.role}")
    print(f"Finish reason:  {response.choices[0].finish_reason}")
    print(f"Prompt tokens:  {response.usage.prompt_tokens}")
    print(f"Output tokens:  {response.usage.completion_tokens}")


# ── Part C: Comparing Models ───────────────────────────────────────────────────

def compare_models() -> None:
    print("\n" + "=" * 60)
    print("PART C: Model Comparison")
    print("=" * 60)

    question = "What is the capital of France? Answer with just the city name."
    models = ["gpt-5-nano", "gpt-5-mini"]

    for model in models:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            # temperature=0.0,
            # max_tokens=20,
        )
        print(f"{model:20s}: {response.choices[0].message.content.strip()!r:15s} "
              f"| {response.usage.total_tokens} tokens")


# ── Part D: Truncation Detection ──────────────────────────────────────────────

def check_truncation() -> None:
    print("\n" + "=" * 60)
    print("PART D: Truncation Detection")
    print("=" * 60)

    # Deliberately low max_tokens to force truncation
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "List 10 countries in Europe, one per line."}],
        temperature=0.0,
        max_tokens=30,  # too short!
    )

    content = response.choices[0].message.content
    finish = response.choices[0].finish_reason

    print(f"Content: {content!r}")
    print(f"Finish reason: {finish}")

    if finish == "length":
        print("WARNING: Response was truncated. Increase max_tokens for complete output.")
    else:
        print("Response completed normally.")


# ── EXERCISES ─────────────────────────────────────────────────────────────────
# TODO: Complete these exercises

def exercise_1a() -> str:
    """
    Make an API call that asks the model to translate
    "Hello, how are you?" into Spanish.
    Return the translated text.
    """
    # YOUR CODE HERE
    raise NotImplementedError


def exercise_1b() -> dict:
    """
    Make an API call with a system prompt that says:
    "You are a pirate. Respond only in pirate speak."
    Ask: "What is 2 + 2?"
    Return a dict with keys: "content", "tokens_used"
    """
    # YOUR CODE HERE
    raise NotImplementedError


def exercise_1c() -> dict:
    """
    Call the API twice with the same question but different temperatures (0.0 and 1.0).
    Question: "Generate a creative name for a coffee shop."
    Return a dict: {"temp_0": "...", "temp_1": "..."}
    Observe: does temperature 0 always return the same answer?
    """
    # YOUR CODE HERE
    raise NotImplementedError


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    first_call()
    inspect_response()
    compare_models()
    check_truncation()

    print("\n" + "=" * 60)
    print("EXERCISES — implement the functions above and test them")
    print("=" * 60)
    try:
        print(f"1a (translation): {exercise_1a()}")
    except NotImplementedError:
        print("1a: Not implemented yet")

    try:
        result = exercise_1b()
        print(f"1b (pirate): {result['content']!r} ({result['tokens_used']} tokens)")
    except NotImplementedError:
        print("1b: Not implemented yet")

    try:
        result = exercise_1c()
        print(f"1c (temp 0.0): {result['temp_0']!r}")
        print(f"1c (temp 1.0): {result['temp_1']!r}")
    except NotImplementedError:
        print("1c: Not implemented yet")
