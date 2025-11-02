# DMAS - Distributed Multi-Agent System

Comparison of long-context vector vs graph memory in distributed LLM-based multi-agent systems.

## Quick Start

Use `dmas.ipynb` for all operations:

1. **Phase 1: Setup** - Configure `.env`, update run id, check ports
2. **Phase 2: Start** - Launch Docker services
3. **Phase 3: Interact** - Load conversations, ask questions
4. **Phase 4: Memory** - Check stats, reset memory
5. **Phase 5: Shutdown** - Stop services

## Model Configuration

All DMAS services now share the lightweight `dmas_llm` adapter and talk directly to the
configured API endpoint. No local Ollama models are required.

- Default chat model: `gpt-4o-mini` (override with the `MODEL` env var per service)
- Default embedding model: `text-embedding-3-small`
- Token accounting and latency are logged automatically via `log_llm_call`

Minimum environment required by each container:

```
USE_API=true
OPENAI_API_KEY=<your-api-key>
API_BASE_URL=https://api.openai.com/v1  # override for OpenRouter/Groq, etc.
MODEL=gpt-4o-mini                       # responder/coordinator specific
EMBED_MODEL=text-embedding-3-small      # memory + embedding usage
MEMORY_BACKEND=<mem0|graphiti>
DMAS_RUN_ID=<free-text experiment id>
```

Because inference happens remotely, container RAM usage drops to the FastAPI + SDK
overhead (≤512MB for coordinator/responder, plus existing Qdrant/Neo4j allocations).

## Shared Adapter Overview

`dmas_llm/` is mounted into every service container. Using it is as simple as:

```python
from dmas_llm.client import LLMClient

llm = LLMClient()
result = llm.chat([{"role": "user", "content": "ping"}], max_tokens=16)
print(result.text)
print(result.usage)
```

The adapter injects metadata (agent name, run id, active memory backend) into the JSON
logs, so you can later answer questions such as “which responder call answered
`run-42` while using the graph backend and how much did it cost?”.

## Environment Cheatsheet

- Add every secret (for example `OPENAI_API_KEY`) manually to `.env`.
- The `dmas.ipynb` notebook only updates `MEMORY_BACKEND`, `DMAS_RUN_ID`, and it inserts placeholders for missing required keys before calling `docker compose`.
