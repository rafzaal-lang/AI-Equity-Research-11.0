# src/services/llm/json_client.py
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

_client: Optional[OpenAI] = None

def _client_instance() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def _safe_json_extract(text: str) -> Dict[str, Any]:
    """Try to extract a JSON object from arbitrary text."""
    if not text:
        return {}
    # strip code fences
    fence = re.search(r"\{.*\}", text, flags=re.S)
    if fence:
        try:
            return json.loads(fence.group(0))
        except Exception:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}

def chat_json(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0.2,
    enforce_json_mode: bool = True,
) -> Dict[str, Any]:
    """
    Ask for JSON. If JSON-mode fails, fall back to plain text and best-effort parse.
    """
    client = _client_instance()
    try:
        kwargs: Dict[str, Any] = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        if enforce_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        content = (r.choices[0].message.content or "").strip()
        return _safe_json_extract(content)
    except Exception:
        # fallback without response_format
        r = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
        content = (r.choices[0].message.content or "").strip()
        return _safe_json_extract(content)
