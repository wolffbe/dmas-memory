import os
import uuid


def get_run_id() -> str:
    return os.getenv("DMAS_RUN_ID", f"local-{uuid.uuid4().hex[:8]}")


def get_memory_backend() -> str:
    return os.getenv("MEMORY_BACKEND", "mem0")


def get_agent_name() -> str:
    return os.getenv("DMAS_AGENT", "unknown-agent")

