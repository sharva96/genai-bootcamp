# openai-gemini-demo

A minimal demo showing how to call **Google Gemini** through the **OpenAI-compatible API**.
Because Gemini exposes an OpenAI-compatible endpoint, you can use the official `openai`
Python client by simply pointing its `base_url` at Gemini.

This project is managed with [`uv`](https://docs.astral.sh/uv/), a fast Python package
and project manager.

## Prerequisites

- Python 3.12 or newer (declared in `.python-version` / `pyproject.toml`)
- A **Gemini API key** — create one for free at https://aistudio.google.com/apikey

---

## 1. Install `uv`

### macOS

```bash
# Option A — official installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Option B — Homebrew
brew install uv
```

After installing, restart your terminal (or run `source ~/.zshrc`) and verify:

```bash
uv --version
```

### Windows

Open **PowerShell** and run:

```powershell
# Option A — official installer (recommended)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Option B — winget
winget install --id=astral-sh.uv -e
```

Close and reopen PowerShell, then verify:

```powershell
uv --version
```

---

## 2. Get the project

Navigate into this project directory.

### macOS

```bash
cd labs/module-04/openai-gemini-demo
```

### Windows (PowerShell)

```powershell
cd labs\module-04\openai-gemini-demo
```

---

## 3. Configure your API key

Copy the template to a real `.env` file, then fill in your Gemini API key.

### macOS

```bash
cp .env.example .env
```

### Windows (PowerShell)

```powershell
Copy-Item .env.example .env
```

Then open `.env` in your editor and set the values:

```ini
GEMINI_API_KEY=your_gemini_api_key_here
BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
MODEL_NAME=gemini-2.0-flash
```

- `GEMINI_API_KEY` — your key from Google AI Studio.
- `BASE_URL` — leave as-is; this is Gemini's OpenAI-compatible endpoint.
- `MODEL_NAME` — any available Gemini model, e.g. `gemini-2.0-flash` or `gemini-2.5-flash`.

> **Note:** `.env` holds your secret key — never commit it. Only `.env.example` (the
> template) belongs in version control.

---

## 4. Install dependencies

`uv sync` reads `pyproject.toml` + `uv.lock` and creates a `.venv` with the exact
pinned versions. Run it from inside the project directory (same command on both
platforms):

```bash
uv sync
```

---

## 5. Run the program

`uv run` automatically uses the project's virtual environment — you do **not** need to
activate it manually.

```bash
uv run python main.py
```

You should see the full API response, the model's answer, and the token usage
(prompt / completion / total).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `uv: command not found` (mac) / `uv is not recognized` (Windows) | Restart your terminal so the updated `PATH` takes effect, or re-run the installer. |
| `401` / authentication error | Check that `GEMINI_API_KEY` in `.env` is correct and has no extra spaces or quotes. |
| `404` model not found | Set `MODEL_NAME` to a valid Gemini model (e.g. `gemini-2.0-flash`). |
| Key not picked up | Confirm the file is named exactly `.env` (not `.env.txt`) and sits in this project folder. |
