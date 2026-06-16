"""
Exercise 5 — Tool-Calling Chatbot
=================================
Run: uv run python exercises/05_tool_calls.py

Goals:
- Give the chatbot real actions: lookup, calculate, take note
- Run the tool-call loop yourself (Chat Completions API)
- Understand when the model decides to call a tool vs. answer directly
"""

import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

SYSTEM_PROMPT = (
    "You are 'Atlas', a travel-planning assistant for SkyTrip Airlines.\n"
    "You have tools for looking up booking status, flight info, and "
    "saving travel notes. Use them when the user asks for live data.\n"
    "When you don't need a tool, just answer directly.\n"
    "Never invent flight times, prices, or booking IDs."
)


# ── Fake data (in real life: a database or API) ───────────────────────────────

FAKE_BOOKINGS = {
    "ABC123": {"flight": "SK412", "name": "Priya Singh", "date": "2026-06-17",
               "from": "LAX", "to": "NRT", "status": "Confirmed"},
    "XYZ789": {"flight": "SK208", "name": "Diego Marques", "date": "2026-07-02",
               "from": "JFK", "to": "LHR", "status": "Cancelled"},
}

FAKE_FLIGHTS = {
    ("SK412", "2026-06-17"): {"dep": "16:35", "arr": "20:10+1", "gate": "C12", "status": "On time"},
    ("SK208", "2026-07-02"): {"dep": "21:55", "arr": "09:30+1", "gate": "—",   "status": "Cancelled"},
}

NOTES: list[dict] = []


# ── Tool implementations ──────────────────────────────────────────────────────

def lookup_booking(pnr: str) -> dict:
    pnr = pnr.upper().strip()
    if pnr not in FAKE_BOOKINGS:
        return {"error": f"No booking found for PNR {pnr!r}"}
    return {"pnr": pnr, **FAKE_BOOKINGS[pnr]}


def lookup_flight(flight_id: str, date: str) -> dict:
    key = (flight_id.upper().strip(), date.strip())
    if key not in FAKE_FLIGHTS:
        return {"error": f"No flight info for {flight_id} on {date}"}
    return {"flight": key[0], "date": key[1], **FAKE_FLIGHTS[key]}


def save_note(text: str) -> dict:
    note = {"id": len(NOTES) + 1, "text": text, "ts": datetime.utcnow().isoformat() + "Z"}
    NOTES.append(note)
    return {"saved": note}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_booking",
            "description": "Look up a booking by 6-character PNR (booking reference).",
            "parameters": {
                "type": "object",
                "properties": {"pnr": {"type": "string", "description": "6-character booking code, e.g. ABC123"}},
                "required": ["pnr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_flight",
            "description": "Live flight status by flight number and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string", "description": "e.g. SK412"},
                    "date":      {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["flight_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a short note for the user about their trip.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

REGISTRY = {
    "lookup_booking": lookup_booking,
    "lookup_flight":  lookup_flight,
    "save_note":      save_note,
}


# ── Tool-call loop ────────────────────────────────────────────────────────────

def run_turn(history: list[dict], user_msg: str, max_iters: int = 4) -> str:
    history.append({"role": "user", "content": user_msg})

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history,
            tools=TOOLS,
            temperature=0.3,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final answer
            history.append({"role": "assistant", "content": msg.content})
            return msg.content

        # The assistant wants to call one or more tools.
        # We must append the assistant tool_call message and each tool result.
        history.append(msg.model_dump(exclude_none=True))
        for call in msg.tool_calls:
            fn_name = call.function.name
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            print(f"   ⤷ tool: {fn_name}({args})")
            try:
                result = REGISTRY[fn_name](**args)
            except Exception as exc:
                result = {"error": str(exc)}
            history.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    return "[stopped after too many tool-call iterations]"


def chat() -> None:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Tool-calling chatbot. Try:")
    print("  - 'Look up booking ABC123'")
    print("  - 'Is flight SK412 on time on June 17?'")
    print("  - 'Remember that I prefer aisle seats'")
    print("  (type 'quit' to exit)\n")

    while True:
        try:
            user_msg = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit", "q"):
            print(f"Saved notes: {NOTES}")
            return
        reply = run_turn(history, user_msg)
        print(f"\nAtlas: {reply}\n")


# ── EXERCISES ─────────────────────────────────────────────────────────────────

def exercise_5a() -> None:
    """
    Add a new tool `convert_currency(amount, from, to)`. Hardcode a tiny FX
    table (USD, EUR, JPY, GBP). Wire it into TOOLS and REGISTRY.
    Test: "How much is $200 USD in yen?"
    """
    raise NotImplementedError


def exercise_5b() -> None:
    """
    Add a safety check: tools that mutate state (like save_note) require
    a user confirmation before executing.
    When the model wants to call save_note, intercept it, print
    "Confirm: save note '...'? [y/n]" and skip the call if the user types n.
    """
    raise NotImplementedError


def exercise_5c() -> None:
    """
    The model sometimes calls lookup_booking with the wrong PNR (e.g. it
    invents one). Add an INSTRUCTION to the system prompt requiring the
    model to ASK the user for the PNR if it doesn't have one, instead of
    guessing. Test that it works.
    """
    raise NotImplementedError


if __name__ == "__main__":
    chat()
