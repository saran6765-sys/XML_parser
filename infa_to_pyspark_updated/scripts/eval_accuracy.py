import os
import json
from typing import Tuple, Optional

# Allow running as a standalone script
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from infa_to_pyspark.agents import extract_code_blocks, extract_sql_sections, validator  # noqa: E402


def load_case(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Load a single test case.

    Expects either:
    - <name>.txt containing the full LLM output with fenced code blocks, and an optional <name>.ast.json
    - or two files: <name>.py and <name>.sql (DML+DDL together) plus <name>.ast.json
    Returns (pyspark_code, sql_combined, ast_json)
    """
    ast_json = None
    base = path.with_suffix("")
    txt = base.with_suffix(".txt")
    py = base.with_suffix(".py")
    sql = base.with_suffix(".sql")
    ast = base.with_suffix(".ast.json")
    if ast.exists():
        ast_json = ast.read_text(encoding="utf-8")
    if txt.exists():
        text = txt.read_text(encoding="utf-8")
        py_code, _ = extract_code_blocks(text)
        ddl, dml = extract_sql_sections(text)
        sql_combined = None
        if ddl or dml:
            parts = []
            if ddl:
                parts.append(ddl)
            if dml:
                parts.append(dml)
            sql_combined = "\n\n".join(parts)
        return py_code, sql_combined, ast_json
    if py.exists() or sql.exists():
        py_code = py.read_text(encoding="utf-8") if py.exists() else None
        sql_combined = sql.read_text(encoding="utf-8") if sql.exists() else None
        return py_code, sql_combined, ast_json
    return None, None, ast_json


def main(folder: str = "tests/eval") -> None:
    p = Path(folder)
    if not p.exists():
        print(f"No eval folder found at {folder}; create test cases to use this harness.")
        return
    cases = sorted(p.glob("*.txt")) + sorted(p.glob("*.py"))  # allow .sql alongside .py
    seen = set()
    unique_cases = []
    for c in cases:
        base = c.with_suffix("").name
        if base in seen:
            continue
        seen.add(base)
        unique_cases.append(c)
    total = 0
    ok = 0
    details = []
    for c in unique_cases:
        py_code, sql_combined, ast_json = load_case(c)
        if py_code is None and sql_combined is None:
            continue
        total += 1
        res = validator(py_code or "", ast_json or None, sql_code=sql_combined)
        passed = isinstance(res, str) and res.strip().lower().startswith("ok:")
        if passed:
            ok += 1
        else:
            details.append((c.name, res))
    if total == 0:
        print("No cases found.")
        return
    acc = ok / total
    print(f"Accuracy@OK: {ok}/{total} = {acc:.2%}")
    if details:
        print("\nFailures:")
        for name, res in details:
            print(f"- {name}:\n{res}\n")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "tests/eval"
    main(folder)
