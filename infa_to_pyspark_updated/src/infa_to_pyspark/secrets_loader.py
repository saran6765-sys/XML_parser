"""Adapter to load secrets from the legacy module location when running locally."""
from __future__ import annotations

import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from secrets_loader import load_secrets_into_env  # type: ignore

__all__ = ["load_secrets_into_env"]

