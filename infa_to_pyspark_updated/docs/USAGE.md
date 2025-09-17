# Run Locally, Architecture, and Agents

## Run Locally (Manual)
- Create and activate a virtual environment
  - Windows (PowerShell)
    - `py -m venv venv`
    - `./venv/Scripts/Activate.ps1`
    - If blocked once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
  - macOS/Linux
    - `python3 -m venv venv`
    - `source venv/bin/activate`
- Install dependencies
  - `python -m pip install --upgrade pip`
  - `python -m pip install -r requirements.txt`
- Configure LLM (choose one)
  - Hosted (OpenAI/Mistral): set API key only (or paste in the sidebar)
  - OpenAI‑compatible local server (LM Studio, vLLM, LiteLLM, etc.)
    - Start your server (e.g., `http://localhost:8000/v1`)
    - Optional `conf/secrets.json`:
      - `{ "API_KEY": "dummy", "API_BASE": "http://localhost:8000/v1" }`
- Run the app
  - `python -m streamlit run streamlit_app.py`

## About The Application
- Converts Informatica mapping XML to PySpark + SQL and validates outputs against XML‑derived logic.
- Unified LLM settings: Model name, API Key, Base URL (OpenAI‑compatible), works with hosted and local models.
- Streaming XML parsing (iterparse) for fast, low‑memory ingest; falls back to standard parsing when needed.
- Reviewer‑guided auto‑fix loop (up to 3 passes, early stop on success) lifts accuracy.
- Tabs UI separates AST, PySpark, SQL, Reviewer, and Notebook; notebook export includes config (ANSI, UTC), helpers, validation.

## Architecture (Pipeline)
1) Extract (streaming) → normalized AST JSON
2) Derive Logic → filters/joins/lookups/routers/aggregations/update strategy
3) Self‑Clarify (agents) → questions + conservative assumptions
4) Generate (LLM) → PySpark + SQL (DDL and DML)
5) Validate (rule‑based) → IO/schema/keys/MERGE/partition checks
6) Review (LLM) → code review + risks/fixes
7) Fix (LLM) → apply reviewer issues; re‑validate; quick re‑review
8) Notebook (optional) → config + helpers + validation + code cells

## Agents And Their Tasks
- Clarifier/Resolver: identify ambiguities and add safe assumptions using AST + intended logic.
- Generator: produce PySpark + SQL from AST + derived logic + assumptions + parameters.
- Validator: rule‑based checks for IO, schema/fields, MERGE, partitioning, surrogate key, best practices.
- Reviewer: senior review of correctness and alignment with AST/logic; produces issues and risks.
- Fixer: applies reviewer issues, updates code, re‑validates, and triggers quick re‑review.

## Notes
- Prefer `python -m streamlit` and `python -m pip` on Windows to avoid shim issues.
- For local models, set Base URL to your OpenAI‑compatible endpoint and a key if required.
