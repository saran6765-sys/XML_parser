"""Adapter to expose transform framework under the package namespace.

Provides access to the original module and its source for notebook export.
"""
from __future__ import annotations

import sys
import pathlib
import importlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tf = importlib.import_module("transform_framework")

# Re-export public names from the original module
from transform_framework import *  # type: ignore  # noqa: F401,F403

# Expose helpers source for notebook embedding
try:
    __helpers_source__ = pathlib.Path(_tf.__file__).read_text(encoding="utf-8")  # type: ignore
except Exception:
    __helpers_source__ = None

__all__ = [name for name in dir(_tf) if not name.startswith("_")]

