"""Adapter module to expose agents without requiring an installed package.

This re-exports from the legacy top-level module `infa_to_pyspark_agents` so
that imports like `from infa_to_pyspark import extractor` continue to work
even when running locally without packaging.
"""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Re-export all agent utilities
from infa_to_pyspark_agents import (  # type: ignore
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

__all__ = [
    "extractor",
    "normalizer",
    "validator",
    "summarize_ast",
    "extract_code_blocks",
    "extract_sql_sections",
    "derive_logic",
    "build_plan",
    "load_few_shots",
    "extractor_streaming",
]

