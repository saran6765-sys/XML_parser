import os
import sys
import pathlib
import json
import time
import streamlit as st
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
try:
    # Ensure src/ is importable when running locally without installation
    ROOT = pathlib.Path(__file__).resolve().parent
    SRC = ROOT / "src"
    if SRC.exists() and str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
except Exception:
    pass

from infa_to_pyspark.secrets_loader import load_secrets_into_env
import httpx

from langchain_openai import ChatOpenAI

from infa_to_pyspark.agents import (
    extractor,
    normalizer,
    validator,
    summarize_ast,
    extract_code_blocks,
    extract_sql_sections,
    derive_logic,
    build_plan,
    load_few_shots,
    extractor_streaming,
)


def _is_mistral_model(name: Optional[str]) -> bool:
    m = (name or "").lower()
    return m.startswith("mistral") or "mixtral" in m or "mistral" in m


def get_llm(api_key: Optional[str], model: Optional[str], base_url: Optional[str] = None):
    # If base_url is provided, prefer OpenAI-compatible ChatOpenAI regardless of model string
    llm_timeout = float(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    llm_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    if api_key:
        # Set both envs for broad compat
        os.environ.setdefault("OPENAI_API_KEY", api_key)
        os.environ.setdefault("MISTRAL_API_KEY", api_key)

    if base_url:
        try:
            return ChatOpenAI(model=model or "gpt-4o", temperature=0, base_url=base_url, timeout=llm_timeout, max_retries=llm_retries)
        except TypeError:
            try:
                return ChatOpenAI(model=model or "gpt-4o", temperature=0, openai_api_base=base_url, timeout=llm_timeout, max_retries=llm_retries)
            except TypeError:
                return ChatOpenAI(model=model or "gpt-4o", temperature=0)

    # No base URL: pick provider by model name
    if _is_mistral_model(model):
        try:
            from langchain_mistralai import ChatMistralAI  # lazy import
        except Exception:
            import streamlit as st
            st.error("Missing dependency: langchain-mistralai. Install: `pip install langchain-mistralai`")
            return None
        endpoint = os.getenv("MISTRAL_BASE_URL") or os.getenv("LLM_BASE_URL")
        try:
            if endpoint:
                return ChatMistralAI(model=model or "mistral-small-latest", temperature=0, endpoint=endpoint, timeout=llm_timeout, max_retries=llm_retries)
            return ChatMistralAI(model=model or "mistral-small-latest", temperature=0, timeout=llm_timeout, max_retries=llm_retries)
        except TypeError:
            if endpoint:
                import streamlit as st
                st.warning("Installed langchain-mistralai does not support custom endpoint; using default.")
            try:
                return ChatMistralAI(model=model or "mistral-small-latest", temperature=0, timeout=llm_timeout, max_retries=llm_retries)
            except TypeError:
                return ChatMistralAI(model=model or "mistral-small-latest", temperature=0)

    # Default: OpenAI
    try:
        return ChatOpenAI(model=model or "gpt-4o", temperature=0, timeout=llm_timeout, max_retries=llm_retries)
    except TypeError:
        return ChatOpenAI(model=model or "gpt-4o", temperature=0)


load_dotenv()
load_secrets_into_env()

st.set_page_config(page_title="Infa XML → PySpark & SQL", layout="wide")
st.title("Informatica XML → PySpark & SQL (with Reviewer)")

with st.sidebar:
    st.header("Settings")
    # Unified LLM settings
    model = st.text_input("Model name", value=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    api_key = st.text_input(
        "API Key",
        value=(os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("MISTRAL_API_KEY") or ""),
        type="password",
        help="Used locally to call your LLM provider or OpenAI-compatible gateway",
    )
    base_url = st.text_input(
        "Base URL (optional)",
        value=(os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("MISTRAL_BASE_URL") or ""),
        help="OpenAI-compatible endpoint (e.g., https://host/v1). Leave blank for provider default.",
    )
    databricks_mode = st.checkbox(
        "Databricks mode (generate notebook)",
        value=True,
        help="Adds a Databricks .py notebook output with cells for Markdown, PySpark, and SQL.",
    )
    optimize_spark = st.checkbox(
        "Optimize Spark (broadcast, prune, repartition)",
        value=True,
        help="Adds optimization guidance to the generation prompt.",
    )
    # Auto-fix passes are handled programmatically (no UI)
    auto_fix_passes = 3
    st.subheader("Parameters")
    src_base = st.text_input("Source base path/DSN (optional)", value="/mnt/raw")
    tgt_db = st.text_input("Target database", value="analytics")
    tgt_table = st.text_input("Target table", value="emp_payroll")
    write_mode = st.selectbox("Write mode", ["overwrite", "append"], index=0)
    shuffle_parts = st.number_input("Shuffle partitions", min_value=1, max_value=1000, value=16, step=1)
    surrogate_key_col = st.text_input("Surrogate key column", value="emp_sk")
    merge_keys = st.text_input("Merge keys (comma-separated)", value="emp_id")
    partition_columns = st.text_input("Partition columns (comma-separated)", value="dept_id")
    enforce_schema = st.checkbox("Enforce target schema before write", value=True)
    merge_fallback = st.checkbox("Allow overwrite fallback if MERGE fails", value=False)
    key_columns = st.text_input("Not-null key columns (comma-separated)", value="emp_id")
    st.caption("Key is only used locally in this session.")

st.markdown(
    "Upload an Informatica mapping XML. The app extracts a simple AST,"
    " generates PySpark code and SQL DDL/query, and runs a reviewer to"
    " validate and summarize the logic."
)

col_left, _ = st.columns([1, 1])

with col_left:
    xml_file = st.file_uploader("Upload Informatica Mapping XML", type=["xml"]) 
    logic_text = st.text_area(
        "Intended Logic (optional)",
        placeholder=(
            "Describe what the mapping should do (filters, joins, aggregates, target).\n"
            "Example: Read SRC_CUSTOMERS, trim name, filter active=1, write TGT_CUSTOMERS."
        ),
        height=120,
    )
    run_btn = st.button("Generate & Review")

# Content tabs for results
tab_ast, tab_py, tab_sql, tab_rev, tab_nb = st.tabs([
    "AST",
    "PySpark",
    "SQL",
    "Reviewer",
    "Notebook",
])
with tab_ast:
    ast_summary_area = st.empty()
with tab_py:
    pyspark_area = st.empty()
with tab_sql:
    sql_area = st.empty()
with tab_rev:
    review_area = st.empty()
with tab_nb:
    notebook_container = st.container()

if run_btn:
    if not xml_file:
        st.error("Please upload an XML file first.")
        st.stop()

    # Extract and normalize AST (streaming/iterparse for large files)
    try:
        # Prefer streaming to reduce memory and speed up parsing
        raw_ast = extractor_streaming(xml_file)
    except Exception:
        # Fallback to full read if streaming fails
        try:
            xml_file.seek(0)
        except Exception:
            pass
        try:
            xml_str = xml_file.read().decode("utf-8", errors="ignore")
        except Exception:
            st.error("Could not read XML file. Ensure it is valid UTF-8 or plain text.")
            st.stop()
        raw_ast = extractor(xml_str)
    norm_ast = normalizer(raw_ast)
    ast_summary_area.code(summarize_ast(norm_ast))

    # Initialize LLMs (generator and reviewer). Use env overrides if provided
    gen_model = os.getenv("GEN_MODEL") or model
    reviewer_model = os.getenv("REVIEW_MODEL") or model
    llm_gen = get_llm(api_key, gen_model, base_url=base_url)
    llm_rev = get_llm(api_key, reviewer_model, base_url=base_url)
    if llm_gen is None or llm_rev is None:
        st.stop()

    # Generation prompt
    logic_json = derive_logic(norm_ast)
    plan_json = build_plan(norm_ast)
    few_shots = load_few_shots(plan_json, folder="few_shots", max_n=2)

    # Backend self-clarification step (agent-to-agent):
    # Clarifier proposes key questions; Resolver proposes conservative assumptions
    # derived from AST and intended logic. We feed assumptions into generation.
    clarify_notes = None
    with st.spinner("Agents clarifying ambiguities from XML..."):
        clarify_prompt = f"""
        Act as two collaborating agents:
        - Clarifier: lists up to 5 critical questions (filters, join keys/types, lookup defaults,
          router thresholds, surrogate key rules, MERGE behavior, partitioning, schema types) that
          materially impact correctness.
        - Resolver: proposes conservative, industry-standard assumptions for each question using
          evidence from the AST and the user's intended logic if present. Prefer no-op or safe
          defaults when uncertain (e.g., inner join, left join for lookups, default 'N' flags,
          row_number() for surrogate key, MERGE on provided keys or primary natural key columns).

        Output strictly as lines of:
        - question -> assumed_answer (short rationale)

        AST (JSON):
        {norm_ast}

        Extracted XML Logic (JSON):
        {logic_json}

        Intended Logic (optional):
        {logic_text or '[none]'}
        """
        try:
            clarify_notes = llm_rev.predict(clarify_prompt)
        except Exception:
            clarify_notes = "- merge keys -> use MERGE_KEYS if provided else inferred natural key (e.g., emp_id)\n- partitioning -> use PARTITION_COLUMNS if provided else none\n- lookup defaults -> left join with null-handling defaults\n- surrogate key -> row_number() over partition if available else global\n- router thresholds -> apply attributes if present else skip routing"

    gen_prompt = f"""
    You are an expert in Informatica → Databricks migration.

    Task:
    Convert this Informatica mapping AST into a production-ready PySpark job and SQL (DDL + DML). Target Databricks/Delta.

    AST (JSON):
    {norm_ast}

    Extracted Logic (from XML; follow exactly):
    {logic_json}

    Plan (intents + slots; implement exactly):
    {plan_json}

    Parameters (must be used; no hardcoded paths):
    - SOURCE_BASE: {src_base}
    - TARGET_DB: {tgt_db}
    - TARGET_TABLE: {tgt_table}
    - WRITE_MODE: {write_mode}
    - SHUFFLE_PARTITIONS: {shuffle_parts}
    - SURROGATE_KEY_COL: {surrogate_key_col}
    - MERGE_KEYS: {merge_keys}
    - PARTITION_COLUMNS: {partition_columns}
    - ENFORCE_SCHEMA: {enforce_schema}
    - MERGE_FALLBACK_OVERWRITE: {merge_fallback}
    - KEY_COLUMNS_NOT_NULL: {key_columns}

    Requirements:
    1. Sources
       - Use spark.read (Delta/Parquet/JDBC/CSV) as per source type.
       - Apply explicit schema.

    2. Transformations
       - Implement EXP_* as withColumn (trim, cast, default values, system timestamp).
       - Implement FIL_* as DataFrame filter() with the exact condition.
       - Implement JNR_* as joins (correct join type, preserve keys + names).
       - Implement LKP_* as lookup joins (broadcast if small) with default handling.
       - Implement AGG_* as groupBy/agg with SUM, COUNT, etc.
       - Implement RTR_* as conditional filters → multiple DataFrames (LOW, MID, HIGH buckets).
       - Implement SEQ_* as surrogate key using row_number() (preferred) or monotonically_increasing_id(), assign to SURROGATE_KEY_COL.
       - Implement UPD_STR as write mode or MERGE INTO (map DD_INSERT, DD_UPDATE, etc.).

    3. Business Logic (from user, optional):
       {logic_text or '[none]'}

    3b. Clarifications/Assumptions (from user Q&A):
       {clarify_notes or '[none]'}

    3c. Few-shot exemplars (style/pattern reference):
    {few_shots or '[none]'}

    4. Targets
       - Write to Delta with saveAsTable.
       - Match target schema exactly (names, types, order).
        - Overwrite or append based on update strategy.
        - Use partitionBy(PARTITION_COLUMNS) on DataFrame writes and PARTITIONED BY in SQL DDL.

    5. SQL
       - Provide TWO fenced SQL blocks:
         a) DDL: CREATE (or CREATE OR REPLACE) TABLE statements for all targets USING DELTA with explicit schema and options.
         b) DML: MERGE INTO (preferred) or INSERT INTO ... SELECT implementing the exact transformation logic (filters, joins, lookups, router thresholds, surrogate keys, update strategy mapping). Use MERGE_KEYS for the ON condition and include DW_OP mapping if relevant.
       - In SQL, prefer row_number() OVER() for surrogate keys. Only use monotonically_increasing_id() if you explicitly note Databricks support.
       - Include fields typically expected in HR/Payroll examples if present: (EMP_SK, EMP_ID, EMP_NAME, DEPT_ID, DEPT_NAME, GROSS_SAL, TAX_RATE, TAX_AMT, NET_SAL, HIRE_DATE, ACTIVE_FLG, LOAD_TS, DW_OP).

    6. Validation
       - Print schema and row counts for sources and targets.
       - Enforce target schema (cast/select) before write when ENFORCE_SCHEMA is true.
       - Validate not-null on KEY_COLUMNS_NOT_NULL and uniqueness of SURROGATE_KEY_COL.
       - In SQL, add checks for null keys and duplicate surrogate keys (group by having count>1).

    Output:
    - Full PySpark job in fenced ```python code block.
    - SQL DDL in fenced ```sql code block.
    - SQL DML (MERGE/INSERT) in a second fenced ```sql code block.
    - Code must be clean, idiomatic, and runnable in Databricks.
    """
    if optimize_spark:
        gen_prompt += "\nPerformance: Use broadcast() for small lookup tables, select only required columns early, cache/persist only when reused 2+ times, enable AQE, and repartition before write to a reasonable number (use SHUFFLE_PARTITIONS). Avoid small files.\n"

    # N-best generation + validator voting (agent-side)
    def _generate_once(p: str) -> str:
        try:
            return llm_gen.predict(p)
        except httpx.HTTPStatusError as e:
            msg = str(e)
            if "service_tier_capacity_exceeded" in msg or "429" in msg:
                st.warning("Model capacity exceeded. Trying a smaller model...")
                fb_model = None
                if _is_mistral_model(gen_model) and gen_model != "mistral-small-latest":
                    fb_model = "mistral-small-latest"
                elif "gpt-4o" in (gen_model or "") and "mini" not in (gen_model or ""):
                    fb_model = "gpt-4o-mini"
                if fb_model:
                    llm_fallback = get_llm(api_key, fb_model, base_url=base_url)
                    time.sleep(0.5)
                    return llm_fallback.predict(p)
                st.error("Provider is at capacity. Try again later or choose another model.")
                st.stop()
            raise

    n_best = int(os.getenv("N_BEST", "3"))
    candidates = []
    with st.spinner(f"Generating {n_best} candidate(s) and voting..."):
        for i in range(max(1, n_best)):
            variant_note = f"\n\nVariant {i+1}: If possible, produce an equivalent but independent implementation."
            out_text = _generate_once(gen_prompt + variant_note)
            py_code, _ = extract_code_blocks(out_text)
            sql_ddl_cand, sql_dml_cand = extract_sql_sections(out_text)
            combined_sql_cand = "\n\n".join([s for s in [sql_ddl_cand, sql_dml_cand] if s]) if (sql_ddl_cand or sql_dml_cand) else None
            val_text = validator(
                py_code or out_text,
                norm_ast,
                sql_code=combined_sql_cand,
                intended_logic=logic_text,
                extra_target_names=[tgt_table.lower(), f"{tgt_db.lower()}.{tgt_table.lower()}"]
            )
            # Simple scoring: OK -> 100; otherwise penalize per listed issue/note line
            vt = (val_text or "").strip().lower()
            if vt.startswith("ok:"):
                score = 100
            else:
                lines = [l for l in (val_text or "").splitlines() if l.strip().startswith("-")]
                score = max(0, 100 - 5 * len(lines))
            candidates.append({
                "out": out_text,
                "py": py_code,
                "ddl": sql_ddl_cand,
                "dml": sql_dml_cand,
                "val": val_text,
                "score": score,
            })
    # Pick best candidate
    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0] if candidates else {"out": "", "py": None, "ddl": None, "dml": None, "val": ""}
    gen_out = best["out"]

    pyspark_code, _ = extract_code_blocks(gen_out)
    sql_ddl, sql_dml = extract_sql_sections(gen_out)

    # Post-process DDL to enforce PARTITIONED BY from parameters if missing
    def _ensure_ddl_partitioning(ddl_text: Optional[str], part_cols: str) -> Optional[str]:
        if not ddl_text:
            return ddl_text
        cols = [c.strip() for c in (part_cols or "").split(",") if c.strip()]
        if not cols:
            return ddl_text
        ddl_lc = ddl_text.lower()
        if "partitioned by" in ddl_lc:
            return ddl_text
        # naive inject after USING DELTA or end of CREATE
        try:
            import re
            m = re.search(r"(create\s+(?:or\s+replace\s+)?table\b[^;]*?using\s+delta)", ddl_lc, flags=re.IGNORECASE)
            if m:
                start = m.end(1)
                injection = f"\nPARTITIONED BY ({', '.join(cols)})"
                return ddl_text[:start] + injection + ddl_text[start:]
        except Exception:
            pass
        return ddl_text

    sql_ddl = _ensure_ddl_partitioning(sql_ddl, partition_columns)
    if pyspark_code:
        pyspark_area.code(pyspark_code, language="python")
    else:
        pyspark_area.write("No PySpark code block found.")

    if sql_ddl or sql_dml:
        if sql_ddl:
            sql_area.write("-- SQL DDL")
            sql_area.code(sql_ddl, language="sql")
        if sql_dml:
            sql_area.write("-- SQL DML (MERGE/INSERT)")
            sql_area.code(sql_dml, language="sql")
    else:
        sql_area.write("No SQL code blocks found.")

    # Rule-based validation
    combined_sql = "\n\n".join([s for s in [sql_ddl, sql_dml] if s]) if (sql_ddl or sql_dml) else None
    extra_targets = [tgt_table.lower(), f"{tgt_db.lower()}.{tgt_table.lower()}"]
    # Use best candidate's validation if N-best ran
    augmented_logic = f"{logic_text or ''}\nMERGE_KEYS={merge_keys}\nPARTITION_COLUMNS={partition_columns}"
    basics = best.get("val") if "best" in locals() and best.get("val") else validator(
        pyspark_code or gen_out,
        norm_ast,
        sql_code=combined_sql,
        intended_logic=augmented_logic,
        extra_target_names=extra_targets,
    )

    # Optional: LLM-guided auto-fix loop to align with XML logic and validator tips
    def _run_predict(prompt_text: str, use_reviewer: bool = True) -> str:
        client = llm_rev if use_reviewer else llm_gen
        mname = reviewer_model if use_reviewer else gen_model
        try:
            return client.predict(prompt_text)
        except httpx.HTTPStatusError as e:
            msg = str(e)
            if "service_tier_capacity_exceeded" in msg or "429" in msg:
                fb_model = None
                if _is_mistral_model(mname) and mname != "mistral-small-latest":
                    fb_model = "mistral-small-latest"
                elif "gpt-4o" in (mname or "") and "mini" not in (mname or ""):
                    fb_model = "gpt-4o-mini"
                if fb_model:
                    llm_fallback = get_llm(api_key, fb_model, base_url=base_url)
                    time.sleep(0.5)
                    return llm_fallback.predict(prompt_text)
            raise

    for i in range(int(auto_fix_passes)):
        fix_prompt = f"""
        You are a meticulous code reviewer and editor for Informatica → Databricks migrations.

        Task:
        - Validate that the generated PySpark and SQL exactly implement the Extracted XML Logic.
        - Enforce that code matches this Plan (intents + slots):
          {plan_json}
        - Use the rule-based validator feedback to fix concrete issues.
        - Keep target/column names and parameters intact. No hardcoded paths.
        - Return corrected code if any mismatch is found. If correct, re-output the same code.

        AST (JSON):
        {norm_ast}

        Extracted XML Logic (JSON):
        {logic_json}

        Intended Logic (optional):
        {logic_text or '[none]'}

        Current PySpark Code:
        ```python
        {pyspark_code or '[missing]'}
        ```

        Current SQL DDL:
        ```sql
        {sql_ddl or '[ddl missing]'}
        ```

        Current SQL DML:
        ```sql
        {sql_dml or '[dml missing]'}
        ```

        Validator Feedback:
        {basics}

        Output strictly in three fenced blocks in this order:
        1) ```python (full PySpark job)```
        2) ```sql (DDL)```
        3) ```sql (DML)```
        """
        with st.spinner(f"Reviewer auto-fix pass {i+1}..."):
            fix_out = _run_predict(fix_prompt, use_reviewer=True)
        new_py, _ = extract_code_blocks(fix_out)
        new_ddl, new_dml = extract_sql_sections(fix_out)
        # If we got updated blocks, adopt them; otherwise keep prior
        pyspark_code = new_py or pyspark_code
        sql_ddl = new_ddl or sql_ddl
        sql_dml = new_dml or sql_dml
        # Re-render
        if pyspark_code:
            pyspark_area.code(pyspark_code, language="python")
        if sql_ddl or sql_dml:
            sql_area.empty()
            if sql_ddl:
                sql_area.write("-- SQL DDL")
                sql_area.code(sql_ddl, language="sql")
            if sql_dml:
                sql_area.write("-- SQL DML (MERGE/INSERT)")
                sql_area.code(sql_dml, language="sql")
        # Re-validate
        combined_sql = "\n\n".join([s for s in [sql_ddl, sql_dml] if s]) if (sql_ddl or sql_dml) else None
        basics = validator(
            pyspark_code or fix_out,
            norm_ast,
            sql_code=combined_sql,
            intended_logic=f"{logic_text or ''}\nMERGE_KEYS={merge_keys}\nPARTITION_COLUMNS={partition_columns}",
            extra_target_names=extra_targets,
        )
        # Early stop if validator passes
        if isinstance(basics, str) and basics.strip().lower().startswith("ok:"):
            break

    # Reviewer LLM prompt
    review_prompt = f"""
    You are a senior data engineering reviewer.
    Goals:
    1) Summarize the mapping in clear terms from the AST
    2) Review the PySpark and SQL for correctness, readability, and alignment with the AST
    3) Verify whether the code satisfies this intended logic (if provided): {logic_text or '[none]'}
    4) Cross-check strictly against the Extracted XML Logic to ensure filters/joins/aggregations/routers/update strategy are implemented.
    5) Flag any risks or missing pieces, and suggest minimal fixes

    AST (JSON):
    {norm_ast}

    Extracted XML Logic (JSON):
    {logic_json}

    Plan (intents + slots; source of truth for implementation):
    {plan_json}

    At the end, add a final fenced JSON block (no prose) named Issues with an array of objects:
    ```json
    [{{"type":"merge_key","msg":"..."}},{{"type":"partition","msg":"..."}}]
    ```

    PySpark Code:
    ```python
    {pyspark_code or '[missing]'}
    ```

    SQL Code (DDL then DML):
    ```sql
    {sql_ddl or '[ddl missing]'}
    ```

    ```sql
    {sql_dml or '[dml missing]'}
    ```

    Respond with:
    - XML Summary: 2-5 sentences
    - Code Review: bullets
    - Logic Check: pass/fail with justification
    - Risks & Fixes: bullets
    """
    with st.spinner("Running reviewer agent..."):
        try:
            review_text = llm_rev.predict(review_prompt)
        except httpx.HTTPStatusError as e:
            msg = str(e)
            if "service_tier_capacity_exceeded" in msg or "429" in msg:
                fb_model = None
                if _is_mistral_model(reviewer_model) and reviewer_model != "mistral-small-latest":
                    fb_model = "mistral-small-latest"
                elif "gpt-4o" in (reviewer_model or "") and "mini" not in (reviewer_model or ""):
                    fb_model = "gpt-4o-mini"
                if fb_model:
                    llm_fallback = get_llm(api_key, fb_model, base_url=base_url)
                    time.sleep(0.5)
                    review_text = llm_fallback.predict(review_prompt)
                else:
                    review_text = "[Reviewer skipped due to capacity limits. Try again later or choose another model.]"
            else:
                raise

    # Extract structured issues if present (fenced json block)
    import re, json as _json
    issues_json = None
    try:
        m = re.search(r"```json\s*(\[.*?\])\s*```", review_text, flags=re.DOTALL)
        if m:
            issues_json = m.group(1)
    except Exception:
        issues_json = None

    # Post-review auto-fix: use Code Review issues and validator notes to correct code
    final_review_text = review_text
    try:
        review_has_issues = review_text and ("Code Review" in review_text or "Risks" in review_text or "Issue" in review_text)
    except Exception:
        review_has_issues = True

    if review_has_issues:
        post_fix_prompt = f"""
        You are a code-fixing agent. Apply the following Code Review findings and Risks & Fixes
        to correct the PySpark and SQL so they strictly implement the AST and Extracted XML Logic.
        Keep parameterization and target/schema contracts intact. Prefer minimal changes.

        AST (JSON):
        {norm_ast}

        Extracted XML Logic (JSON):
        {logic_json}

        Intended Logic (optional):
        {logic_text or '[none]'}

        Current PySpark Code:
        ```python
        {pyspark_code or '[missing]'}
        ```

        Current SQL DDL:
        ```sql
        {sql_ddl or '[ddl missing]'}
        ```

        Current SQL DML:
        ```sql
        {sql_dml or '[dml missing]'}
        ```

        Validator Feedback:
        {basics}

        Code Review Findings:
        {review_text}

        Issues JSON (machine-readable):
        {issues_json or '[]'}

        Output strictly in three fenced blocks in this order:
        1) ```python (full PySpark job)```
        2) ```sql (DDL)```
        3) ```sql (DML)```
        """
        with st.spinner("Applying fixes from Code Review..."):
            fix2_out = _run_predict(post_fix_prompt, use_reviewer=True)
        _new_py, _ = extract_code_blocks(fix2_out)
        _new_ddl, _new_dml = extract_sql_sections(fix2_out)
        if _new_py or _new_ddl or _new_dml:
            pyspark_code = _new_py or pyspark_code
            sql_ddl = _new_ddl or sql_ddl
            sql_dml = _new_dml or sql_dml
            if pyspark_code:
                pyspark_area.code(pyspark_code, language="python")
            if sql_ddl or sql_dml:
                sql_area.empty()
                if sql_ddl:
                    sql_area.write("-- SQL DDL")
                    sql_area.code(sql_ddl, language="sql")
                if sql_dml:
                    sql_area.write("-- SQL DML (MERGE/INSERT)")
                    sql_area.code(sql_dml, language="sql")
            # Re-validate
            combined_sql = "\n\n".join([s for s in [sql_ddl, sql_dml] if s]) if (sql_ddl or sql_dml) else None
            basics = validator(
                pyspark_code or fix2_out,
                norm_ast,
                sql_code=combined_sql,
                intended_logic=logic_text,
                extra_target_names=extra_targets,
            )
            # Re-review quickly (best-effort)
            review_prompt2 = f"""
            You are a senior data engineering reviewer. Re-check after fixes.
            Goals:
            - Verify PySpark/SQL alignment with AST and Extracted XML Logic
            - Confirm earlier issues are addressed; list any remaining gaps succinctly

            AST (JSON):
            {norm_ast}

            Extracted XML Logic (JSON):
            {logic_json}

            PySpark Code:
            ```python
            {pyspark_code or '[missing]'}
            ```

            SQL Code (DDL then DML):
            ```sql
            {sql_ddl or '[ddl missing]'}
            ```

            ```sql
            {sql_dml or '[dml missing]'}
            ```
            """
            try:
                with st.spinner("Re-running reviewer after fixes..."):
                    final_review_text = llm_rev.predict(review_prompt2)
            except Exception:
                pass

    def _to_magic_block(prefix: str, text: str) -> str:
        lines = (text or "").splitlines() or [""]
        return "\n".join([f"# MAGIC {prefix}"] + [f"# MAGIC {l}" for l in lines])

    def build_databricks_notebook(md_text: str, pyspark: Optional[str], sql_ddl: Optional[str], sql_dml: Optional[str]) -> str:
        parts = ["# Databricks notebook source"]
        # Markdown cell
        parts.append("# COMMAND ----------")
        parts.append(_to_magic_block("%md", md_text.strip()))
        # Helpers cell (transformation framework)
        parts.append("# COMMAND ----------")
        try:
            from infa_to_pyspark import transform_framework as tf
            helpers = getattr(tf, "__helpers_source__", None)
            if not helpers:
                import inspect
                helpers = inspect.getsource(tf)
        except Exception:
            helpers = "# Helpers not bundled"
        parts.append(helpers)
        # Config cell
        parts.append("# COMMAND ----------")
        # Derive expected schema from AST targets (if present)
        try:
            ast_obj = json.loads(norm_ast)
            target_schemas = [
                {
                    "table": (t.get("name") or "target").lower(),
                    "fields": [
                        {"name": (f.get("name") or "").lower(), "type": (f.get("datatype") or "").lower()}
                        for f in (t.get("fields") or [])
                    ],
                }
                for t in (ast_obj.get("targets") or [])
            ]
        except Exception:
            target_schemas = []

        cfg = f"""
spark.conf.set('spark.sql.shuffle.partitions', '{shuffle_parts}')
spark.conf.set('spark.sql.adaptive.enabled', 'true')
spark.conf.set('spark.sql.ansi.enabled', 'true')
spark.conf.set('spark.sql.session.timeZone', 'UTC')

SOURCE_BASE = '{src_base}'
TARGET_DB = '{tgt_db}'
TARGET_TABLE = '{tgt_table}'
WRITE_MODE = '{write_mode}'
TARGET_FQN = f"{tgt_db}.{tgt_table}"
SURROGATE_KEY_COL = '{surrogate_key_col}'
MERGE_KEYS = [{', '.join([repr(k.strip()) for k in merge_keys.split(',') if k.strip()])}]
PARTITION_COLUMNS = [{', '.join([repr(c.strip()) for c in partition_columns.split(',') if c.strip()])}]
ENFORCE_SCHEMA = {str(enforce_schema).lower()}
MERGE_FALLBACK_OVERWRITE = {str(merge_fallback).lower()}
EXPECTED_TARGET_SCHEMAS = {json.dumps(target_schemas)}
KEY_COLUMNS_NOT_NULL = [{', '.join([repr(c.strip()) for c in key_columns.split(',') if c.strip()])}]
"""
        parts.append(cfg)
        # Post-write validation cell
        parts.append("# COMMAND ----------")
        validation_py = """
from transform_framework import (
    validate_table_schema_exact,
    validate_partitioning,
    validate_not_null_df,
    validate_unique_df,
    validate_surrogate_key_df,
)

issues = []
# Validate schema
if ENFORCE_SCHEMA and EXPECTED_TARGET_SCHEMAS:
    first = EXPECTED_TARGET_SCHEMAS[0]
    expected = [(f['name'], f.get('type') or 'string') for f in first.get('fields', [])]
    issues += validate_table_schema_exact(spark, TARGET_FQN, expected)

# Table-level df for further checks
df_target = spark.table(TARGET_FQN)

# Not-null on key columns
issues += validate_not_null_df(df_target, KEY_COLUMNS_NOT_NULL)

# Surrogate key uniqueness
issues += validate_surrogate_key_df(df_target, SURROGATE_KEY_COL)

# Optional: partition validation
issues += validate_partitioning(spark, TARGET_FQN, PARTITION_COLUMNS)

print("Validation issues:")
for i in issues:
    print("-", i)
if not issues:
    print("Validation passed: schema, keys, partitioning")
        """
        parts.append(validation_py)
        # PySpark cell
        parts.append("# COMMAND ----------")
        parts.append(pyspark.strip() if pyspark else "# No PySpark code generated")
        # SQL cell via magic
        parts.append("# COMMAND ----------")
        sql_combined = []
        if sql_ddl:
            sql_combined.append("-- DDL\n" + sql_ddl.strip())
        if sql_dml:
            sql_combined.append("-- DML\n" + sql_dml.strip())
        sql_text = "\n\n".join(sql_combined) if sql_combined else "-- No SQL code generated"
        parts.append(_to_magic_block("%sql", sql_text))
        return "\n".join(parts) + "\n"

    if databricks_mode:
        md = """### Mapping Summary and Intended Logic

AST Summary:
{ast_summary}

Intended Logic:
{logic}
""".format(ast_summary=summarize_ast(norm_ast), logic=logic_text or "[none provided]")

        notebook_py = build_databricks_notebook(md, pyspark_code, sql_ddl, sql_dml)
        with tab_nb:
            st.code(notebook_py, language="python")
            st.download_button(
                "Download .py notebook",
                notebook_py,
                file_name="infa_mapping_notebook.py",
                mime="text/x-python",
            )

    # Show reviewer output (LLM) + our basic checks
    with tab_rev:
        review_area.markdown(final_review_text)
        st.info(basics)
        if issues_json:
            try:
                parsed = _json.loads(issues_json)
                if isinstance(parsed, list) and parsed:
                    st.write("Issues (structured):")
                    for it in parsed:
                        t = it.get("type", "issue")
                        msg = it.get("msg", "")
                        st.code(f"[{t}] {msg}")
            except Exception:
                pass

