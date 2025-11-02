import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict


def _default_serializer(obj: Any):
    """Best-effort serializer so complex objects don't break logging."""

    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return str(obj)
    except Exception:
        return str(obj)


def log_llm_call(**data: Dict[str, Any]) -> None:
    """Emit a single structured JSON log line for an LLM call."""

    payload = dict(data)
    payload["ts"] = datetime.now(timezone.utc).isoformat()

    sys.stdout.write(json.dumps(payload, default=_default_serializer) + "\n")
    sys.stdout.flush()

