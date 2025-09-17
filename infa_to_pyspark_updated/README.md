# Informatica XML → PySpark/SQL (with Reviewer)

Generate production-oriented PySpark and SQL (DDL + DML) from Informatica mapping XML, then validate the output against XML‑derived logic using an LLM “code reviewer” and a rule‑based validator. Optionally, export a Databricks `.py` notebook that includes validation cells.

## Features
- XML to AST: Extracts sources, targets, transformations, fields, and lineage from Informatica XML.
- Streaming XML parsing: Uses iterparse-based streaming extractor for large XML files to reduce memory and improve load speed; falls back to regular parsing if needed.
- Logic derivation: Heuristically derives filters, joins, lookups, routers, and aggregations from XML metadata.
- Code generation: Prompts an LLM to produce PySpark and SQL (DDL + DML) targeting Databricks/Delta.
- Rule‑based validator: Checks for IO, schema/target coverage, MERGE/partitioning practices, surrogate key, and basic best practices.
- LLM reviewer: Cross‑checks generated code against the AST, derived logic, and optional intended logic; summarizes risks and fixes.
- Auto‑fix loop: Runs reviewer‑guided fixes for N passes; re‑validates and updates the final code blocks.
- Databricks notebook: Builds a downloadable `.py` notebook with config, helpers, validation cells, PySpark, and SQL.
- Provider choice: OpenAI or Mistral via LangChain, selectable in the sidebar.
- New helpers for migration accuracy:
  - `lookup_asof()` nearest-prior/next lookups
  - `apply_lookup_range(..., include_min, include_max)` bound control
  - `allocate_sequence_range()` global sequence allocation (Delta)
  - `write_audit_log()` simple audit trail per run
  - `add_delta_check_constraints()` to enforce invariants

## Quick Start
1) Create and activate a virtual environment
- Windows (PowerShell):
  - `py -m venv venv`
  - `./venv/Scripts/Activate.ps1`
  - If execution is blocked: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- macOS/Linux:
  - `python3 -m venv venv`
  - `source venv/bin/activate`

2) Install dependencies
- `python -m pip install --upgrade pip`
- `python -m pip install -r requirements.txt`

3) Set an API key and optional Base URL
- OpenAI: set `OPENAI_API_KEY` (shell or `.env`), or paste in the Streamlit sidebar.
- Mistral: set `MISTRAL_API_KEY` (shell or `.env`), or paste in the sidebar.
- Custom Base URL (proxy/self-host): set `OPENAI_BASE_URL` / `MISTRAL_BASE_URL` or the generic `LLM_BASE_URL`; you can also enter it in the sidebar.
- Optional: put credentials in `conf/secrets.json` with keys `API_KEY` and `API_BASE` and the app will load them:
  - Example `conf/secrets.json`:
    ```json
    { "API_KEY": "sk-...", "API_BASE": "https://your-gateway.example.com/v1" }
    ```

4) Run the app
- Prefer the module form to avoid Windows shim issues:
  - `python -m streamlit run streamlit_app.py`

If you see a Windows “Fatal error in launcher”, use `python -m streamlit ...` and `python -m pip ...` as shown above. If problems persist, recreate the venv.

## Using the App
- Upload an Informatica mapping XML.
- Optionally enter “Intended Logic” to guide validation where XML is sparse.
- LLM settings (unified, OpenAI‑compatible):
  - Model name: e.g., `gpt-4o-mini`, `gpt-4o`, `mistral-small-latest`, `open-mixtral-8x7b`, or a local model id.
  - API Key: your provider key (or a dummy value for local OpenAI‑compatible gateways).
  - Base URL (optional): your OpenAI‑compatible endpoint (e.g., `http://localhost:8000/v1`). Leave blank to use the provider’s default.
- Other toggles:
  - Databricks notebook export
  - Optimize Spark
  - LLM auto‑fix passes (0–3)
- Click “Generate & Review”.

You’ll see:
- AST Summary
- Generated PySpark and SQL blocks
- Reviewer Findings (LLM)
- Rule‑based validation notes
- Optional Databricks notebook content + download button

## Configuration
- Environment variables can be set in `.env` or your shell:
  - `OPENAI_API_KEY`, `MISTRAL_API_KEY`
  - Optional: `OPENAI_BASE_URL`, `MISTRAL_BASE_URL`, or generic `LLM_BASE_URL`
  - Optional defaults: `LLM_PROVIDER`, `LLM_MODEL`
- You can also paste keys into the sidebar fields; these are used locally for this session.

### Auto-Fix Passes
- The app runs up to 3 reviewer-guided auto-fix passes by default (no front-end control) and stops early once the rule-based validator reports OK.

## How It Works (Key Modules)
- `streamlit_app.py`
  - UI, parameter capture, and orchestration of extract → derive → self‑clarify (agents) → generate → validate → review → auto‑fix.
  - Builds prompts and renders results; exports a `.py` Databricks notebook with validation cells.
- `infa_to_pyspark_agents.py`
  - `extractor(xml_str)` parses Informatica XML into a structured AST.
  - `normalizer(ast_json)` lowercases names and normalizes structure.
  - `derive_logic(ast_json)` heuristically derives filters/joins/lookups/routers/aggregations.
  - `validator(pyspark_code, ast_json, sql_code, intended_logic, extra_target_names)` checks IO, schema coverage, MERGE/partitioning, surrogate key, best practices, and consistency hints.
  - `summarize_ast(ast_json)` produces a human summary for the UI.
  - `extract_code_blocks(text)` and `extract_sql_sections(text)` pull fenced code from LLM responses.
- `transform_framework.py`
  - Reusable helpers for expressions, filters, joins, lookups (range/as‑of), routers, aggregations, surrogate key, global sequence allocation, Delta MERGE, audit logging, constraints, and validation (schema/partitioning/not‑null/unique/SK checks).

Code references (open these to explore):
- `streamlit_app.py:160` generation prompt with AST + derived logic and parameters.
- `streamlit_app.py:270` rule‑based validation call and data flow.
- `streamlit_app.py:300` reviewer‑guided auto‑fix loop and re‑validation.
- `infa_to_pyspark_agents.py:1` XML extractor and utilities.
- `transform_framework.py:1` PySpark/Delta helpers and runtime validations.

## Outputs
- PySpark code block: a Databricks‑ready transformation job using parameters.
- SQL DDL block: `CREATE TABLE ... USING DELTA` with schema and options.
- SQL DML block: `MERGE INTO` (preferred) or `INSERT INTO ... SELECT` matching the XML logic.
- Databricks `.py` notebook: Markdown summary, helpers, config, validation cell, PySpark cell, `%sql` cell.
  - Config now enables ANSI mode and sets session timezone to UTC.

## Troubleshooting
- Windows “Fatal error in launcher” for `pip`/`streamlit`:
  - Use `python -m pip ...` and `python -m streamlit ...`.
  - If still broken, recreate the venv: deactivate, delete `venv`, then `py -m venv venv` and reinstall.
- Model capacity errors (429/service tier capacity):
  - The app automatically retries with a smaller model for Mistral where possible. Otherwise, try again or switch provider/model.
- Cannot activate venv on PowerShell:
  - Run once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

## Notes
- Do not commit real API keys. `.env` is for local convenience only.
- Prefer OpenAI GPT‑4o or Mistral Large for best results; if you hit capacity, switch to smaller models or reduce auto‑fix passes.

## Accuracy and Fine‑Tuning
- Prompt/Controls (no training required)
  - Provide clear “Intended Logic” (used by agents when resolving ambiguities).
  - The backend agents auto‑clarify and run up to 3 reviewer‑guided fixes; no front‑end control required.
  - Tune models: try `gpt-4o`/`gpt-4o-mini` or `mistral-large-latest`/`mistral-small-latest` for cost/latency tradeoffs.
- Data‑level: add a few labeled exemplars of XML→PySpark/SQL as in‑context few‑shots (contact us if you want a loader hook).
- OpenAI fine‑tune (example; check your account availability):
  1. Prepare `train.jsonl` with chat examples: one object per line like `{ "messages": [{"role":"system","content":"..."},{"role":"user","content":"AST..."},{"role":"assistant","content":"```python...```\n```sql...```"}] }`.
  2. Upload + create job (CLI):
     - `openai files create -p training -f train.jsonl`
     - `openai finetunes create -t <file_id> -m gpt-4o-mini`
  3. Use the returned fine‑tuned model id in the app’s model field.
- Mistral fine‑tune (if enabled for your org):
  - Use their SDK/console to create a supervised fine‑tune on `mistral-small`/`open-mixtral-8x7b` with your JSONL chats, then set the returned model in the sidebar.
- If you host your own model behind a proxy (vLLM/TGI), point the app at the proxy via `LLM_BASE_URL` and use an OpenAI‑compatible route; then consider LoRA/adapters for domain adaptation.
