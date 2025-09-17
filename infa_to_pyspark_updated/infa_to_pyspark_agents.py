import os
import json
import re
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, IO, List, Dict
from dotenv import load_dotenv
from secrets_loader import load_secrets_into_env

from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI


def extractor(xml_str: str) -> str:
    """Extract a richer AST from Informatica XML.

    Captures:
    - sources/targets with fields and datatypes
    - transformations with type, table attributes, and transformfields
    - connectors (lineage)
    """
    root = ET.fromstring(xml_str)
    mapping_name = root.attrib.get("NAME", "unknown_mapping")

    def _fields(parent):
        out = []
        for f in parent.findall(".//SOURCEFIELD") + parent.findall(".//TARGETFIELD"):
            out.append({
                "name": f.attrib.get("NAME"),
                "datatype": f.attrib.get("DATATYPE"),
                "precision": f.attrib.get("PRECISION"),
                "scale": f.attrib.get("SCALE"),
            })
        return out

    sources = []
    for s in root.findall(".//SOURCE"):
        sources.append({
            "name": s.attrib.get("NAME"),
            "fields": _fields(s),
        })

    targets = []
    for t in root.findall(".//TARGET"):
        targets.append({
            "name": t.attrib.get("NAME"),
            "fields": _fields(t),
        })

    transformations = []
    for tr in root.findall(".//TRANSFORMATION"):
        t_attrs = {}
        for ta in tr.findall(".//TABLEATTRIBUTE"):
            k = ta.attrib.get("NAME") or ta.attrib.get("name")
            v = ta.attrib.get("VALUE") or ta.attrib.get("value")
            if k:
                t_attrs[k] = v
        t_fields = []
        for tf in tr.findall(".//TRANSFORMFIELD"):
            t_fields.append({
                "name": tf.attrib.get("NAME"),
                "datatype": tf.attrib.get("DATATYPE"),
                "precision": tf.attrib.get("PRECISION"),
                "scale": tf.attrib.get("SCALE"),
                "expression": tf.attrib.get("EXPRESSION") or tf.attrib.get("EXPR") or tf.attrib.get("DEFAULTVALUE"),
            })
        transformations.append({
            "name": tr.attrib.get("NAME"),
            "type": tr.attrib.get("TYPE"),
            "table_attributes": t_attrs,
            "fields": t_fields,
        })

    connectors = []
    for c in root.findall(".//CONNECTOR"):
        connectors.append({
            "from_instance": c.attrib.get("FROMINSTANCE"),
            "from_field": c.attrib.get("FROMFIELD"),
            "to_instance": c.attrib.get("TOINSTANCE"),
            "to_field": c.attrib.get("TOFIELD"),
        })

    ast = {
        "mapping_name": mapping_name,
        "sources": sources,
        "targets": targets,
        "transformations": transformations,
        "connectors": connectors,
    }
    return json.dumps(ast, indent=2)


def extractor_streaming(xml_file: IO[bytes]) -> str:
    """Streaming extractor using iterparse to support large XML files.

    Processes the XML incrementally and clears elements after use to
    keep memory low. Accepts a file-like object opened in binary mode
    (e.g., Streamlit UploadedFile).
    """
    from xml.etree.ElementTree import iterparse

    def _tag(t: str) -> str:
        # Strip any namespace like {uri}TAG
        return t.split('}')[-1] if '}' in t else t

    # Defaults
    mapping_name = "unknown_mapping"
    sources = []
    targets = []
    transformations = []
    connectors = []

    # Ensure file pointer at start
    try:
        xml_file.seek(0)
    except Exception:
        pass

    for event, elem in iterparse(xml_file, events=("start", "end")):
        tag = _tag(elem.tag)
        if event == "start":
            if tag == "MAPPING":
                mapping_name = elem.attrib.get("NAME", mapping_name)
            continue

        # end event
        if tag == "SOURCE":
            # collect fields under this SOURCE
            fields = []
            for f in elem.findall(".//SOURCEFIELD"):
                fields.append({
                    "name": f.attrib.get("NAME"),
                    "datatype": f.attrib.get("DATATYPE"),
                    "precision": f.attrib.get("PRECISION"),
                    "scale": f.attrib.get("SCALE"),
                })
            sources.append({
                "name": elem.attrib.get("NAME"),
                "fields": fields,
            })
            elem.clear()
        elif tag == "TARGET":
            fields = []
            for f in elem.findall(".//TARGETFIELD"):
                fields.append({
                    "name": f.attrib.get("NAME"),
                    "datatype": f.attrib.get("DATATYPE"),
                    "precision": f.attrib.get("PRECISION"),
                    "scale": f.attrib.get("SCALE"),
                })
            targets.append({
                "name": elem.attrib.get("NAME"),
                "fields": fields,
            })
            elem.clear()
        elif tag == "TRANSFORMATION":
            t_attrs = {}
            for ta in elem.findall(".//TABLEATTRIBUTE"):
                k = ta.attrib.get("NAME") or ta.attrib.get("name")
                v = ta.attrib.get("VALUE") or ta.attrib.get("value")
                if k:
                    t_attrs[k] = v
            t_fields = []
            for tf in elem.findall(".//TRANSFORMFIELD"):
                t_fields.append({
                    "name": tf.attrib.get("NAME"),
                    "datatype": tf.attrib.get("DATATYPE"),
                    "precision": tf.attrib.get("PRECISION"),
                    "scale": tf.attrib.get("SCALE"),
                    "expression": tf.attrib.get("EXPRESSION") or tf.attrib.get("EXPR") or tf.attrib.get("DEFAULTVALUE"),
                })
            transformations.append({
                "name": elem.attrib.get("NAME"),
                "type": elem.attrib.get("TYPE"),
                "table_attributes": t_attrs,
                "fields": t_fields,
            })
            elem.clear()
        elif tag == "CONNECTOR":
            connectors.append({
                "from_instance": elem.attrib.get("FROMINSTANCE"),
                "from_field": elem.attrib.get("FROMFIELD"),
                "to_instance": elem.attrib.get("TOINSTANCE"),
                "to_field": elem.attrib.get("TOFIELD"),
            })
            elem.clear()

    ast = {
        "mapping_name": mapping_name,
        "sources": sources,
        "targets": targets,
        "transformations": transformations,
        "connectors": connectors,
    }
    return json.dumps(ast, indent=2)

def normalizer(raw_ast: str) -> str:
    ast = json.loads(raw_ast)
    # Lowercase names while preserving structure
    def _lc_name(x):
        if isinstance(x, dict):
            if "name" in x and isinstance(x["name"], str):
                x["name"] = x["name"].lower()
            if "fields" in x and isinstance(x["fields"], list):
                for f in x["fields"]:
                    if isinstance(f, dict) and "name" in f and isinstance(f["name"], str):
                        f["name"] = f["name"].lower()
            if "table_attributes" in x and isinstance(x["table_attributes"], dict):
                # keep keys as-is; values stay original
                pass
        return x

    ast["sources"] = [_lc_name(s) for s in ast.get("sources", [])]
    ast["targets"] = [_lc_name(t) for t in ast.get("targets", [])]
    ast["transformations"] = [_lc_name(tr) for tr in ast.get("transformations", [])]
    return json.dumps(ast, indent=2)


def derive_logic(ast_json: str) -> str:
    """Derive key logic snippets (filters, joins, lookups, router thresholds) from AST.

    This is heuristic and depends on presence of TABLEATTRIBUTE/TRANSFORMFIELD metadata.
    """
    try:
        ast = json.loads(ast_json)
    except Exception:
        return json.dumps({}, indent=2)

    logic = {"filters": [], "joins": [], "lookups": [], "routers": [], "aggregations": [], "update_strategy": []}
    for tr in ast.get("transformations", []) or []:
        name = tr.get("name")
        ttype = (tr.get("type") or "").lower()
        attrs = tr.get("table_attributes") or {}
        fields = tr.get("fields") or []
        # Generic extraction of conditions/expressions
        for k, v in attrs.items():
            kv = (k or "").lower()
            if v:
                if "cond" in kv or "filter" in kv:
                    logic["filters"].append({"transformation": name, "condition": v})
                if "join" in kv:
                    logic["joins"].append({"transformation": name, "detail": f"{k}: {v}"})
                if "lookup" in kv or "lkp" in kv:
                    logic["lookups"].append({"transformation": name, "detail": f"{k}: {v}"})
                if "group" in kv or "agg" in kv:
                    logic["aggregations"].append({"transformation": name, "detail": f"{k}: {v}"})
        for f in fields:
            expr = f.get("expression")
            if expr:
                logic.setdefault("expressions", []).append({"transformation": name, "field": f.get("name"), "expr": expr})
        # Heuristic by type
        if ttype.startswith("filter"):
            if attrs:
                logic["filters"].append({"transformation": name, "attrs": attrs})
        if ttype.startswith("router"):
            logic["routers"].append({"transformation": name, "attrs": attrs})
        if ttype.startswith("join"):
            logic["joins"].append({"transformation": name, "attrs": attrs})
        if "lookup" in ttype:
            logic["lookups"].append({"transformation": name, "attrs": attrs})
        if ttype.startswith("agg"):
            logic["aggregations"].append({"transformation": name, "attrs": attrs})
        if "update" in ttype:
            logic["update_strategy"].append({"transformation": name, "attrs": attrs})

    return json.dumps(logic, indent=2)


def build_plan(ast_json: str) -> str:
    """Build a compact Plan object (intents + slots) from AST.

    The Plan summarizes what to implement, reducing ambiguity during generation:
    {
      "intents": ["read", "filter", "join:left", "lookup", "aggregate", "router", "sequence", "merge"],
      "slots": {"filters": [...], "joins": [...], "lookups": [...], "router": [...],
                 "sequence": {"method": "row_number", "partition": [...], "order": [...]},
                 "merge_keys": [...], "partition_columns": [...], "target": "db.tbl"}
    }
    """
    try:
        ast = json.loads(ast_json)
    except Exception:
        return json.dumps({"intents": [], "slots": {}}, indent=2)

    intents: List[str] = ["read"]
    slots: Dict[str, object] = {}
    # Filters
    filters: List[Dict[str, str]] = []
    # Joins
    joins: List[Dict[str, str]] = []
    # Lookups
    lookups: List[Dict[str, str]] = []
    # Aggregations
    aggs: List[Dict[str, str]] = []
    # Router
    routers: List[Dict[str, str]] = []

    for tr in ast.get("transformations", []) or []:
        ttype = (tr.get("type") or "").lower()
        attrs = tr.get("table_attributes") or {}
        if ttype.startswith("filter") or any("cond" in (k or "").lower() for k in attrs.keys()):
            intents.append("filter")
            for k, v in attrs.items():
                if "cond" in (k or "").lower() or "filter" in (k or "").lower():
                    filters.append({"name": tr.get("name"), "condition": v})
        if "join" in ttype:
            intents.append("join:left" if "left" in ttype else ("join:right" if "right" in ttype else "join:inner"))
            joins.append({"name": tr.get("name"), "type": ttype, "attrs": attrs})
        if "lookup" in ttype:
            intents.append("lookup")
            lookups.append({"name": tr.get("name"), "attrs": attrs})
        if ttype.startswith("agg"):
            intents.append("aggregate")
            aggs.append({"name": tr.get("name"), "attrs": attrs})
        if ttype.startswith("router"):
            intents.append("router")
            routers.append({"name": tr.get("name"), "attrs": attrs})
        if "update" in ttype:
            intents.append("merge")

    if filters:
        slots["filters"] = filters
    if joins:
        slots["joins"] = joins
    if lookups:
        slots["lookups"] = lookups
    if routers:
        slots["router"] = routers
    if aggs:
        slots["aggregations"] = aggs

    # target
    tgt = None
    if ast.get("targets"):
        tgt = (ast["targets"][0].get("name") or "target").lower()
    if tgt:
        slots["target_name"] = tgt

    # Attempt to infer sequence
    slots["sequence"] = {"method": "row_number", "partition": [], "order": []}

    return json.dumps({"intents": sorted(list(set(intents))), "slots": slots}, indent=2)


def load_few_shots(plan_json: str, folder: str = "few_shots", max_n: int = 2) -> str:
    """Load a few-shot examples from folder.

    Each file may be .json with fields: {"title", "user", "assistant"} where user/assistant are text.
    Selection heuristic: pick examples whose filename or metadata matches plan intents (join, lookup, router, agg).
    Returns a formatted string to be appended to the prompt.
    """
    intents = []
    try:
        intents = (json.loads(plan_json) or {}).get("intents") or []
    except Exception:
        pass
    wanted = set([i.split(":")[0] for i in intents])
    paths: List[str] = []
    try:
        import os as _os
        if _os.path.isdir(folder):
            for fn in sorted(_os.listdir(folder)):
                if not fn.lower().endswith(".json"):
                    continue
                tag = fn.lower()
                if any(k in tag for k in ["join", "lookup", "router", "agg", "scd", "merge"]) and (
                    not wanted or any(k in tag for k in wanted)
                ):
                    paths.append(_os.path.join(folder, fn))
                if len(paths) >= max_n:
                    break
    except Exception:
        return ""

    shots: List[str] = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                ex = json.load(f)
            title = ex.get("title") or p
            user = ex.get("user") or ""
            assistant = ex.get("assistant") or ""
            shots.append(f"Example: {title}\nUser:\n{user}\nAssistant:\n{assistant}")
        except Exception:
            continue
    return "\n\n".join(shots)


def validator(
    pyspark_code: str,
    ast_json: Optional[str] = None,
    sql_code: Optional[str] = None,
    intended_logic: Optional[str] = None,
    extra_target_names: Optional[list] = None,
) -> str:
    """Lightweight validator for generated PySpark code.

    - Checks for reads/writes
    - Optionally validates presence of key transform/target names from AST
    - Heuristics for Informatica → Databricks rules and schema coverage
    """
    issues = []
    tips = []

    py = (pyspark_code or "").lower()
    sql_lc = (sql_code or "").lower()

    # Extract hints from intended_logic (MERGE_KEYS, PARTITION_COLUMNS)
    merge_keys_hint: list[str] = []
    part_cols_hint: list[str] = []
    try:
        text = intended_logic or ""
        for line in text.splitlines():
            if line.strip().upper().startswith("MERGE_KEYS="):
                val = line.split("=", 1)[1]
                merge_keys_hint = [k.strip().lower() for k in val.split(",") if k.strip()]
            if line.strip().upper().startswith("PARTITION_COLUMNS="):
                val = line.split("=", 1)[1]
                part_cols_hint = [c.strip().lower() for c in val.split(",") if c.strip()]
    except Exception:
        pass

    # Basic IO checks
    if "spark.read" not in py:
        issues.append("Missing spark.read source load")
    if all(tok not in py for tok in [".write", "saveastable"]):
        issues.append("Missing DataFrame write/saveAsTable")

    # AST presence checks
    if ast_json:
        try:
            ast = json.loads(ast_json)
            # Accept both old (list[str]) and new (list[dict{name}]) shapes
            def _names(seq):
                out = []
                for x in seq or []:
                    if isinstance(x, str):
                        out.append(x.lower())
                    elif isinstance(x, dict) and x.get("name"):
                        out.append(str(x["name"]).lower())
                return out

            tgt_names = _names(ast.get("targets", []))
            # Allow parameterized targets (e.g., TARGET_DB.TARGET_TABLE) to satisfy checks
            if extra_target_names:
                for n in extra_target_names:
                    if isinstance(n, str):
                        tgt_names.append(n.lower())
            xforms = _names(ast.get("transformations") or ast.get("transforms", []))
            if tgt_names and not any(t in py for t in tgt_names):
                issues.append(
                    "Target names not referenced in PySpark: " + ", ".join(tgt_names)
                )
            if tgt_names and sql_code is not None and not any(t in sql_lc for t in tgt_names):
                issues.append(
                    "Target names not referenced in SQL: " + ", ".join(tgt_names)
                )
            missing_xforms = [x for x in xforms if x and x not in py]
            if missing_xforms and len(xforms) <= 25:
                tips.append(
                    f"Transforms not obviously implemented in PySpark: {', '.join(missing_xforms)}"
                )
        except Exception:
            tips.append("Could not parse AST for deeper checks")

    # Transformation Accuracy heuristics
    # EXP_CLEAN defaults & timestamp
    if any(tag in py for tag in ["exp_clean", "exp_cleaner", "exp_"]):
        if "trim(" not in py and "rtrim(" not in py:
            tips.append("EXP_* present but no trim() found")
        if "withcolumn(" in py and ("gross_sal" in py or "active_flg" in py):
            pass
        else:
            tips.append("Defaults for gross_sal=0 and active_flg='n' not detected")
        if "current_timestamp()" not in py and "current_timestamp()" not in sql_lc:
            tips.append("LOAD_TS not set via current_timestamp()")

    # FIL_ACTIVE should use active_flg not is_active
    if "filter(" in py:
        if "is_active" in py and "active_flg" not in py:
            issues.append("Filter uses is_active; expected active_flg")

    # Router thresholds (salary buckets)
    if "rtr_" in py or "router" in py or "bucket" in py:
        if not ("25000" in py and "50000" in py):
            tips.append("Router thresholds 25k/50k not found; verify bucket logic")

    # EXP_TAX: tax_amt and net_sal
    if "tax" in py:
        if "tax_amt" not in py:
            issues.append("tax_amt not computed")
        if "net_sal" not in py:
            issues.append("net_sal not computed")

    # LKP_TAX: salary range join
    if "lkp" in py or "lookup" in py:
        has_range_join = ("slab_min" in py and "slab_max" in py) and (
            ">=" in py and "<" in py or "between" in py
        )
        if not has_range_join:
            tips.append("Lookup join does not reference slab_min/slab_max range")

    # Target schema coverage (prefer AST target fields if present; fallback to common HR list)
    required_cols: list[str] = []
    try:
        if ast_json:
            ast_obj = json.loads(ast_json)
            targets = ast_obj.get("targets") or []
            if isinstance(targets, list) and targets:
                t0 = targets[0]
                fields = t0.get("fields") if isinstance(t0, dict) else None
                if fields:
                    required_cols = [str((f.get("name") or "").lower()) for f in fields if isinstance(f, dict)]
    except Exception:
        pass
    if not required_cols:
        required_cols = [
            "emp_sk","emp_id","emp_name","dept_id","dept_name","gross_sal",
            "tax_rate","tax_amt","net_sal","hire_date","active_flg","load_ts","dw_op",
        ]

    if required_cols:
        missing_py_cols = [c for c in required_cols if c and c not in py]
        if missing_py_cols:
            issues.append("PySpark missing target fields: " + ", ".join(missing_py_cols))
        if sql_code is not None:
            missing_sql_cols = [c for c in required_cols if c and c not in sql_lc]
            if missing_sql_cols:
                issues.append("SQL DDL missing target fields: " + ", ".join(missing_sql_cols))

    # Data lineage fidelity heuristics
    if "groupby(" in py and "agg(" in py and all(k not in py for k in ["saveastable", ".write"]):
        tips.append("Aggregation present but not clearly propagated to final target")

    if "merge into" not in sql_lc and "merge(" not in py and ("upd_str" in py or "update strategy" in py):
        tips.append("Update strategy present but no MERGE/insert logic detected")

    # Consistency between PySpark & SQL
    if sql_code:
        if ("dept_name" in sql_lc) and ("dept_name" not in py):
            tips.append("dept_name present in SQL but not in PySpark final select")
        # SQL surrogate key: prefer row_number
        if "monotonically_increasing_id(" in sql_lc:
            tips.append("SQL uses monotonically_increasing_id(); ensure Databricks engine or prefer row_number() OVER()")
        if "row_number() over" not in sql_lc and "monotonically_increasing_id(" not in sql_lc:
            tips.append("No explicit SQL surrogate key function detected; consider row_number() OVER()")
        # DDL partitioning present
        if "partitioned by" not in sql_lc:
            tips.append("SQL DDL missing PARTITIONED BY; add explicit partitioning if required")

    # Extra rules: MERGE parity, partition parity, field coverage
    try:
        ast = json.loads(ast_json) if ast_json else {}
    except Exception:
        ast = {}
    # Extract target fields
    tgt_fields = []
    try:
        for t in (ast.get("targets") or [])[:1]:
            tgt_fields = [str((f.get("name") or "").lower()) for f in (t.get("fields") or [])]
    except Exception:
        pass
    # Field coverage: warn if extra fields absent in code
    if tgt_fields:
        missing_in_py = [c for c in tgt_fields if c and c not in py]
        if missing_in_py and ("issues" not in locals() or f"PySpark missing target fields" not in issues):
            tips.append("target fields may be incomplete in PySpark: " + ", ".join(missing_in_py))
        if sql_code is not None:
            missing_in_sql = [c for c in tgt_fields if c and c not in sql_lc]
            if missing_in_sql:
                tips.append("target fields may be incomplete in SQL: " + ", ".join(missing_in_sql))

    # Partition parity
    part_cols = []
    try:
        # Expect comma-separated list in parameters section or env; we only check presence in code text
        # Heuristic: user input partition_columns was passed earlier
        pass
    except Exception:
        pass
    if "partitionby(" in py and "partitioned by" not in sql_lc:
        tips.append("Partitioning present in PySpark but missing in SQL DDL")
    if "partitioned by" in sql_lc and "partitionby(" not in py:
        tips.append("Partitioning present in SQL DDL but missing in PySpark write")

    # MERGE parity
    if "merge into" in (sql_lc or ""):
        # if merge keys were indicated in prompt, look for 'on' and ensure key tokens appear
        on_idx = sql_lc.find("merge into")
        on_part = sql_lc[on_idx:on_idx + 500]
        for k in [" and ", "=", " on "]:
            if " on " in sql_lc:
                break
        if " on " not in sql_lc:
            tips.append("MERGE lacks ON clause in SQL")


    # Optimization & best practices
    if "broadcast(" not in py and ("lkp" in py or "lookup" in py):
        tips.append("Consider broadcast() for small lookup joins")
    if "repartition(" not in py and (".write" in py or "saveastable" in py):
        tips.append("Consider repartition() before write to avoid small files")
    if "count()" not in py and "printschema" not in py:
        tips.append("Add row count and printSchema() checks for validation")

    # Surrogate key presence
    if "emp_sk" not in py and "surrogate" not in py:
        tips.append("Surrogate key column (e.g., EMP_SK) not detected; add row_number() or monotonically_increasing_id()")
    if "row_number(" not in py and "monotonically_increasing_id(" not in py:
        tips.append("No sequence generator detected (row_number or monotonically_increasing_id)")

    # Update strategy presence (MERGE)
    if "merge into" not in sql_lc and "merge(" not in py and "delta_merge(" not in py:
        tips.append("No update strategy MERGE detected; use Delta MERGE INTO with keys")

    # Strict MERGE key check if hints provided
    if merge_keys_hint and "merge into" in sql_lc:
        on_clause = ""
        try:
            import re as _re
            m = _re.search(r"merge\s+into\b.*?\bon\s+(.*?)\s+when\s+matched", sql_lc, flags=_re.DOTALL)
            if not m:
                m = _re.search(r"merge\s+into\b.*?\bon\s+(.*)$", sql_lc, flags=_re.DOTALL)
            if m:
                on_clause = m.group(1)
        except Exception:
            on_clause = ""
        for k in merge_keys_hint:
            if k and k not in on_clause:
                tips.append(f"MERGE ON clause missing hinted key: {k}")

    # Partitioning presence
    if "partitionby(" not in py and "partitioned by" not in sql_lc:
        tips.append("No explicit partitioning found; use partitionBy() and PARTITIONED BY for Delta tables")

    # Strict partition parity with hints
    if part_cols_hint:
        missing_py_parts = [c for c in part_cols_hint if c not in py]
        missing_sql_parts = [c for c in part_cols_hint if c not in sql_lc]
        if missing_py_parts:
            tips.append("partitionBy missing hinted columns: " + ", ".join(missing_py_parts))
        if missing_sql_parts:
            tips.append("SQL PARTITIONED BY missing hinted columns: " + ", ".join(missing_sql_parts))

    if not issues and not tips:
        return "OK: Code passes basic and Databricks migration checks"

    details = []
    if issues:
        details.append("Issues:\n- " + "\n- ".join(issues))
    if tips:
        details.append("Notes:\n- " + "\n- ".join(tips))
    return "\n\n".join(details)


load_dotenv()
load_secrets_into_env()


def get_llm_from_env():
    """Return an LLM based on env vars.

    LLM_PROVIDER: openai|mistral (default: openai)
    LLM_MODEL: overrides default model for the provider
    """
    # If MISTRAL_API_KEY exists and no explicit provider, prefer mistral
    env_provider = os.getenv("LLM_PROVIDER")
    if env_provider:
        provider = env_provider.lower()
    else:
        provider = "mistral" if os.getenv("MISTRAL_API_KEY") else "openai"
    model = os.getenv("LLM_MODEL")
    if provider == "mistral":
        if not model:
            model = "mistral-large-latest"
        try:
            from langchain_mistralai import ChatMistralAI  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "langchain-mistralai is not installed. Install with: pip install langchain-mistralai"
            ) from e
        endpoint = os.getenv("MISTRAL_BASE_URL") or os.getenv("LLM_BASE_URL")
        timeout = float(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT_SECONDS", "60")))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        try:
            if endpoint:
                return ChatMistralAI(model=model, temperature=0, endpoint=endpoint, timeout=timeout, max_retries=max_retries)
            return ChatMistralAI(model=model, temperature=0, timeout=timeout, max_retries=max_retries)
        except TypeError:
            # Backwards compatibility
            try:
                return ChatMistralAI(model=model, temperature=0, timeout=timeout, max_retries=max_retries)
            except TypeError:
                return ChatMistralAI(model=model, temperature=0)
    # default to OpenAI
    if not model:
        model = "gpt-4o"
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    timeout = float(os.getenv("LLM_TIMEOUT", os.getenv("LLM_TIMEOUT_SECONDS", "60")))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    try:
        if base:
            return ChatOpenAI(model=model, temperature=0, base_url=base, timeout=timeout, max_retries=max_retries)
        return ChatOpenAI(model=model, temperature=0, timeout=timeout, max_retries=max_retries)
    except TypeError:
        if base:
            try:
                return ChatOpenAI(model=model, temperature=0, openai_api_base=base, timeout=timeout, max_retries=max_retries)
            except TypeError:
                pass
        return ChatOpenAI(model=model, temperature=0)


# Initialize a base LLM instance (optional; streamlit app may create its own)
llm = get_llm_from_env()

tools = [
    Tool(
        name="Extractor",
        func=extractor,
        description="Extract AST from Informatica XML",
    ),
    Tool(
        name="Normalizer",
        func=normalizer,
        description="Normalize AST to canonical JSON format",
    ),
    Tool(
        name="Validator",
        func=validator,
        description="Check generated PySpark code for issues",
    ),
]

agent = initialize_agent(
    tools, llm, agent="zero-shot-react-description", verbose=True
)


def summarize_ast(ast_json: str) -> str:
    try:
        ast = json.loads(ast_json)
    except Exception:
        return "Could not parse AST"
    mapping = ast.get("mapping_name", "<unknown>")
    sources = ast.get("sources", [])
    targets = ast.get("targets", [])
    transforms = ast.get("transformations", [])
    parts = [f"Mapping: {mapping}"]
    parts.append(
        "Sources: "
        + (", ".join([s.get("name", "?") or "?" for s in sources]) if sources else "-")
    )
    parts.append(
        "Targets: "
        + (", ".join([t.get("name", "?") or "?" for t in targets]) if targets else "-")
    )
    if transforms:
        parts.append(
            "Transforms: "
            + ", ".join(
                [
                    (tr.get("name") or "?") + (f"[{tr.get('type')}]" if tr.get("type") else "")
                    for tr in transforms
                ]
            )
        )
    # brief counts
    parts.append(f"Source fields total: {sum(len(s.get('fields', [])) for s in sources)}")
    parts.append(f"Target fields total: {sum(len(t.get('fields', [])) for t in targets)}")
    return "\n".join(parts)


def extract_code_blocks(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract PySpark and SQL code blocks from LLM text output."""
    if not text:
        return None, None
    # Simple heuristics to find fenced code blocks
    pyspark_code = None
    sql_code = None

    # Prefer blocks labeled as python/pyspark and sql
    code_blocks = re.findall(r"```(\w+)?\n(.*?)```", text, flags=re.DOTALL)
    for lang, block in code_blocks:
        lang_lc = (lang or "").lower()
        if pyspark_code is None and lang_lc in {"python", "pyspark"}:
            pyspark_code = block.strip()
        if sql_code is None and lang_lc == "sql":
            sql_code = block.strip()

    # Heuristic classification for unlabeled blocks
    if (pyspark_code is None or sql_code is None) and code_blocks:
        for lang, block in code_blocks:
            lang_lc = (lang or "").lower()
            b_lc = block.lower()
            if pyspark_code is None and not lang_lc and (
                "from pyspark" in b_lc or "spark." in b_lc or "sparkSession" in block
            ):
                pyspark_code = block.strip()
            if sql_code is None and not lang_lc and (
                "create table" in b_lc or "merge into" in b_lc or "insert into" in b_lc or b_lc.strip().startswith("select")
            ):
                sql_code = block.strip()

    # Fallback: try to split sections by headings
    if pyspark_code is None:
        m = re.search(r"(?is)\b(pyspark|python).*?```.*?```", text)
        if m:
            inner = re.search(r"```.*?\n(.*?)```", m.group(0), re.DOTALL)
            if inner:
                pyspark_code = inner.group(1).strip()

    if sql_code is None:
        m = re.search(r"(?is)\bsql\b.*?```.*?```", text)
        if m:
            inner = re.search(r"```.*?\n(.*?)```", m.group(0), re.DOTALL)
            if inner:
                sql_code = inner.group(1).strip()

    return pyspark_code, sql_code


def extract_sql_sections(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract SQL DDL and SQL DML/Query as two sections, if present.

    Heuristics:
    - Classify fenced ```sql blocks: those containing CREATE TABLE are DDL; those with MERGE/INSERT/SELECT are DML.
    - If only one sql block, return it as DML and leave DDL None unless it contains CREATE.
    """
    _, sql_block = extract_code_blocks(text)
    ddl = None
    dml = None
    if not text:
        return None, None
    # Find all sql code blocks
    blocks = re.findall(r"```sql\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for b in blocks:
        b_lc = b.lower()
        if any(kw in b_lc for kw in ["create table", "create or replace table", "ddl"]):
            if ddl is None:
                ddl = b.strip()
        if any(kw in b_lc for kw in ["merge into", "insert into", "select", "update "]):
            # Prefer MERGE/INSERT as DML; plain SELECT as fallback
            if dml is None:
                dml = b.strip()
    # Fallback to single sql block
    if not blocks and sql_block:
        dml = sql_block
    # Heuristic: extract from unlabeled code blocks
    if ddl is None or dml is None:
        unlabeled = re.findall(r"```\n(.*?)```", text, flags=re.DOTALL)
        for b in unlabeled:
            b_lc = b.lower()
            if ddl is None and ("create table" in b_lc or "create or replace table" in b_lc):
                ddl = b.strip()
            if dml is None and ("merge into" in b_lc or "insert into" in b_lc or b_lc.strip().startswith("select")):
                dml = b.strip()
    # Last resort: scan raw text for SQL keywords
    if ddl is None and re.search(r"(?is)create\s+(or\s+replace\s+)?table", text or ""):
        m = re.search(r"(?is)(create\s+(?:or\s+replace\s+)?table.*?)(?:\n\n|$)", text)
        if m:
            ddl = m.group(1).strip()
    if dml is None and re.search(r"(?is)merge\s+into|insert\s+into|^\s*select", text or ""):
        m = re.search(r"(?is)((?:merge\s+into|insert\s+into|select).*?)(?:\n\n|$)", text)
        if m:
            dml = m.group(1).strip()
    return ddl, dml


if __name__ == "__main__":
    xml_str = """<MAPPING NAME="m_customer_load">
        <SOURCE NAME="SRC_CUSTOMERS"/>
        <TARGET NAME="TGT_CUSTOMERS"/>
        <TRANSFORMATION NAME="EXP_CLEANUP"/>
    </MAPPING>"""

    print("=== Running Pipeline with Agents ===\n")

    raw_ast = extractor(xml_str)
    print("AST:\n", raw_ast, "\n")

    norm_ast = normalizer(raw_ast)
    print("Normalized AST:\n", norm_ast, "\n")

    prompt = f"""
    You are an expert in Informatica → Databricks migration.

    Task:
    Convert this Informatica mapping AST into a production-ready PySpark job and SQL DDL.

    AST:
    {norm_ast}

    Requirements:
    1. **Sources**
       - Use spark.read (Delta/Parquet/JDBC/CSV) as per source type.
       - Apply explicit schema.

    2. **Transformations**
       - Implement EXP_* as withColumn (trim, cast, default values, system timestamp).
       - Implement FIL_* as DataFrame filter() with the exact condition.
       - Implement JNR_* as joins (correct join type, preserve keys + names).
       - Implement LKP_* as lookup joins (broadcast if small) with default handling.
       - Implement AGG_* as groupBy/agg with SUM, COUNT, etc.
       - Implement RTR_* as conditional filters → multiple DataFrames (LOW, MID, HIGH buckets).
       - Implement SEQ_* as surrogate key using monotonically_increasing_id() or row_number().
       - Implement UPD_STR as write mode or MERGE INTO (map DD_INSERT, DD_UPDATE, etc.).

    3. **Business Logic**
       - TAX_AMT = GROSS_SAL * TAX_RATE / 100
       - NET_SAL = GROSS_SAL - TAX_AMT
       - ACTIVE_FLG = default 'N', filter where ACTIVE_FLG = 'Y'

    4. **Targets**
       - Write to Delta with saveAsTable.
       - Match target schema exactly (names, types, order).
       - Overwrite or append based on update strategy.

    5. **SQL DDL**
       - Generate CREATE TABLE statements for all targets.
       - Ensure all fields (EMP_SK, EMP_ID, EMP_NAME, DEPT_ID, DEPT_NAME, GROSS_SAL, TAX_RATE, TAX_AMT, NET_SAL, HIRE_DATE, ACTIVE_FLG, LOAD_TS, DW_OP).

    6. **Validation**
       - Print schema and row counts for sources and targets.

    Output:
    - Full PySpark job in fenced ```python code block.
    - SQL DDLs in fenced ```sql code block.
    - Code must be clean, idiomatic, and runnable in Databricks.
    """
    code_output = llm.predict(prompt)
    print("Generated Code:\n", code_output, "\n")

    validation = validator(code_output, norm_ast)
    print("Validation:\n", validation)
