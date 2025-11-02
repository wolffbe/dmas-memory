import os
import time
from typing import Any, Dict, Iterable, NamedTuple

from openai import OpenAI

from .ids import get_agent_name, get_memory_backend, get_run_id
from .logging import log_llm_call


class LLMResult(NamedTuple):
    """Standard return object for LLMClient invocations."""

    text: str
    usage: Dict[str, int]
    message: Dict[str, Any]
    raw: Dict[str, Any]

    def __iter__(self) -> Iterable:
        # Allow unpacking like: text, usage = llm.chat(...)
        yield self.text
        yield self.usage
        yield self.message
        yield self.raw


class LLMClient:
    """Small adapter that standardises access to chat and embedding models."""

    def __init__(self) -> None:
        self.use_api = os.getenv("USE_API", "false").lower() == "true"
        self.agent = get_agent_name()

        if self.use_api:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY environment variable is required when USE_API=true")

            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
            )
            self.chat_model = os.getenv("MODEL", "gpt-4o-mini")
            self.embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
        else:
            # Placeholder for local inference backends (e.g. Ollama) if re-introduced later.
            self.client = None
            self.chat_model = os.getenv("MODEL", "llama3.1:8b")
            self.embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")

    def chat(self, messages, model: str | None = None, **kwargs) -> LLMResult:
        """Execute a chat completion request and emit structured logs."""

        run_id = get_run_id()
        memory_backend = get_memory_backend()
        t0 = time.time()
        target_model = model or self.chat_model

        if self.use_api:
            response = self.client.chat.completions.create(
                model=target_model,
                messages=messages,
                **kwargs,
            )

            message = response.choices[0].message
            text = message.content or ""
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            message_dict = message.model_dump()
            raw_dict = response.model_dump()
        else:
            raise NotImplementedError("Local inference backend is not implemented in LLMClient")

        latency = time.time() - t0

        log_llm_call(
            run_id=run_id,
            agent=self.agent,
            memory_backend=memory_backend,
            model=target_model,
            messages=messages,
            response=text,
            usage=usage,
            latency=latency,
            project=getattr(self, "project", None),
        )

        return LLMResult(text=text, usage=usage, message=message_dict, raw=raw_dict)

    def embed(self, text: str):
        if not self.use_api:
            raise NotImplementedError("Local embedding backend is not implemented in LLMClient")

        response = self.client.embeddings.create(
            model=self.embed_model,
            input=text,
        )

        return response.data[0].embedding

