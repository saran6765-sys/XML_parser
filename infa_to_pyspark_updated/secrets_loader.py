import json
import os
from typing import Optional


def load_secrets_into_env(
    path_candidates: Optional[list] = None,
) -> None:
    """Load API key and base URL from a JSON secrets file into environment.

    Looks for keys: API_KEY, API_BASE (or BASE_URL). Does not overwrite
    existing env vars. Intended for simple local setups.
    """
    candidates = path_candidates or [
        os.path.join("conf", "secrets.json"),
        "secrets.json",
    ]
    for p in candidates:
        try:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except Exception:
            continue

        api_key = data.get("API_KEY") or data.get("api_key")
        api_base = (
            data.get("API_BASE")
            or data.get("BASE_URL")
            or data.get("api_base")
            or data.get("base_url")
        )

        # Populate generic env vars if not already set
        if api_key and not os.getenv("LLM_API_KEY"):
            os.environ["LLM_API_KEY"] = str(api_key)
        if api_key and not os.getenv("OPENAI_API_KEY") and not os.getenv("MISTRAL_API_KEY"):
            # set OpenAI key by default; user can still override in UI
            os.environ["OPENAI_API_KEY"] = str(api_key)
        if api_base and not os.getenv("LLM_BASE_URL"):
            os.environ["LLM_BASE_URL"] = str(api_base)
        # Also set provider-specific base if not present
        if api_base and not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = str(api_base)
        if api_base and not os.getenv("MISTRAL_BASE_URL"):
            os.environ["MISTRAL_BASE_URL"] = str(api_base)
        break

