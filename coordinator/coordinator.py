import json
import os
import uuid
import threading
from typing import Optional, Dict, Any
import re
from datetime import datetime, timedelta

import requests
import tiktoken
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dmas_llm.client import LLMClient


app = FastAPI(title="coordinator", version="1.0")

# ---------------------------
# JOB STORE IN RAM
# ---------------------------
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

def job_log(job_id: str, msg: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.setdefault("logs", []).append(msg)

def _get_tokenizer():
    encoding_name = os.getenv("TIKTOKEN_ENCODING", "cl100k_base")
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:  # noqa: BLE001
        return tiktoken.get_encoding("cl100k_base")


TOKENIZER = _get_tokenizer()


def count_tokens(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(TOKENIZER.encode(text))


class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    limit: Optional[int] = None
    difficulty: Optional[str] = "auto"


class PreviewRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None
    limit: Optional[int] = None


class Coordinator:    
    def __init__(self, locomo_url: str, memory_url: str, responder_url: str):
        self.locomo_url = locomo_url
        self.memory_url = memory_url
        self.responder_url = responder_url
        self.context = None
        self.llm = LLMClient()
   
    def _do_load_conversation_job(self, job_id: str, conv_index: int):
        """Function that runs in a separate thread: does the long work and updates the status."""
        
        # job log
        job_log(job_id, f"starting load for conversation {conv_index}")

        # set job status to running
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "running"
        
        # 1) get conversation data from locomo
        try:
            locomo_resp = requests.get(
                f"{self.locomo_url}/conversations/index/{conv_index}",
                timeout=30
            )
            locomo_resp.raise_for_status()
            conversation_data = locomo_resp.json()
            sample_id = conversation_data.get("sample_id")
            with JOBS_LOCK:
                JOBS[job_id]["sample_id"] = sample_id
            job_log(job_id, f"fetched from locomo (sample_id={sample_id})")
        except Exception as e:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = f"locomo error: {e}"
            job_log(job_id, f"locomo error: {e}")
            return

        # 2) send conversation data to memory (ASYNC)
        mem_payload = {
            "conv_index": conv_index,
            "data": conversation_data,
        }

        try:
            # memory now responds immediately with {status: started, job_id: ...}
            mem_resp = requests.post(
                f"{self.memory_url}/memorize",
                json=mem_payload,
                timeout=10,  # it's short because it's only the start
            )
            mem_resp.raise_for_status()
            mem_result = mem_resp.json()
            mem_job_id = mem_result.get("job_id")
            if not mem_job_id:
                # fallback, in case memory was not async
                job_log(job_id, "memory returned no job_id, assuming done")
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "done"
                return

            with JOBS_LOCK:
                JOBS[job_id]["memory_job_id"] = mem_job_id
            
            job_log(job_id, f"memory job started ({mem_job_id})")
        except Exception as e:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = f"memory start error: {e}"
            job_log(job_id, f"memory start error: {e}")
            return

        # 3) poll memory
        # we do it here, inside the coordinator thread
        import time
        POLL_INTERVAL = 3
        MAX_WAIT_S = 900  # 15 minutes, overkill but safe
        start_t = time.time()
        last_len = 0

        while True:
            if time.time() - start_t > MAX_WAIT_S:
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = "error"
                    JOBS[job_id]["error"] = "memory job timeout"
                job_log(job_id, "memory job timeout")
                return

            try:
                st_resp = requests.get(
                    f"{self.memory_url}/memorize/status/{mem_job_id}",
                    timeout=10,
                )
                st_resp.raise_for_status()
                st_data = st_resp.json()
            except Exception as e:
                job_log(job_id, f"polling memory failed: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            # copy memory logs to coordinator
            mem_logs = st_data.get("logs") or []
            # print only new logs
            for line in mem_logs[last_len:]:
                job_log(job_id, f"[memory] {line}")
            last_len = len(mem_logs)

            mem_status = st_data.get("status")
            if mem_status in ("done", "error"):
                # also bring up the count
                with JOBS_LOCK:
                    JOBS[job_id]["status"] = mem_status
                    if "memories_added" in st_data:
                        JOBS[job_id]["memories_added"] = st_data["memories_added"]
                    if "error" in st_data:
                        JOBS[job_id]["error"] = st_data["error"]
                job_log(job_id, f"memory finished with status={mem_status}")
                return

            time.sleep(POLL_INTERVAL)


    # async / start load
    def start_load_conversation(self, conv_index: int) -> dict:
        job_id = str(uuid.uuid4())
        with JOBS_LOCK:
            JOBS[job_id] = {
                "status": "pending",
                "conversation_index": conv_index,
                "logs": [],
            }

        # start thread
        t = threading.Thread(
            target=self._do_load_conversation_job,
            args=(job_id, conv_index),
            daemon=True,
        )
        t.start()

        return {
            "status": "started",
            "job_id": job_id,
            "conversation_index": conv_index,
        }

    def get_job_status(self, job_id: str) -> dict:
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return {"status": "error", "error": "job not found"}
            # shallow copy so we don't modify the original
            return dict(job)
   
    def load_all_conversations(self) -> dict:
        results = []
        for conv_idx in range(10):
            started = self.start_load_conversation(conv_idx)
            results.append(started)
        return {
            "status": "started",
            "total": 10,
            "results": results,
        }
    
    def _define_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Query memory for relevant information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "What to ask memory"
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "respond",
                    "description": "Send question and context to responder for final answer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The user's question"
                            },
                            "context": {
                                "type": "string",
                                "description": "Context from memory"
                            }
                        },
                        "required": ["question", "context"]
                    }
                }
            }
        ]
    
    def _call_tool(self, tool_name: str, arguments: dict) -> dict:
        if tool_name == "remember":
            prompt = arguments["prompt"]
            try:
                response = requests.post(
                    f"{self.memory_url}/query",
                    json={"prompt": prompt},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "error":
                    return {"error": result.get("error")}
                
                context = result.get("context", "")
                
                if context:
                    self.context = context
                    return {"context": "cached"}
                else:
                    self.context = ""
                    return {"context": "none"}
                
            except Exception as e:
                return {"error": str(e)}
        
        elif tool_name == "respond":
            question = arguments["question"]
            try:
                response = requests.post(
                    f"{self.responder_url}/respond",
                    json={
                        "question": question,
                        "context": self.context
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                if result.get("status") == "error":
                    return {"error": result.get("error")}
                
                return {"answer": result.get("answer")}
            except Exception as e:
                return {"error": str(e)}
        
        return {"error": "Unknown tool"}
    
    def _memory_query(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"prompt": question}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if limit is not None:
            payload["limit"] = limit

        response = requests.post(
            f"{self.memory_url}/query",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    FACTOID_PREFIXES = (
        "when", "what date", "which day", "at what time",
        "when did", "when was"
    )

    EVENT_VERBS = (
        "attended", "joined", "went to", "went", "participated in",
        "took part in", "took part", "was at"
    )

    def _is_factoid(self, question: str) -> bool:
        q = question.strip().lower()
        return q.startswith(self.FACTOID_PREFIXES)

    def _parse_locomo_ts(self, ts: str) -> Optional[datetime]:
        if not ts:
            return None
        for fmt in ("%I:%M %p on %d %B, %Y", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
        return None

    def _text_has_event_verb(self, text: str) -> bool:
        tlow = text.lower()
        return any(v in tlow for v in self.EVENT_VERBS)

    def _build_inferred_events_block(self, question: str, items: list) -> str:
        if not self._is_factoid(question):
            return ""

        inferred_lines = []
        for it in items:
            md = it.get("metadata") or {}
            ts = md.get("timestamp")
            text = it.get("text") or it.get("formatted") or ""
            if not text or not ts:
                continue

            if not self._text_has_event_verb(text):
                continue

            dt = self._parse_locomo_ts(ts)
            if not dt:
                continue

            # general rule: the event may have happened before
            event_dt = dt - timedelta(days=1)
            inferred_lines.append(
                f"- inferred event date: {event_dt.strftime('%-d %B %Y')} ← from session {ts} saying: “{text}”"
            )

        if not inferred_lines:
            return ""

        header = "INFERRED EVENTS (conversation may report past events):\n"
        return header + "\n".join(inferred_lines) + "\n\n"

    def preview_context(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Preview raw vs condensed context for a given question.
        
        This endpoint demonstrates the RAG pipeline's context compression:
        1. Query memory service to retrieve relevant conversation chunks
        2. Show raw context token count (what memory returns)
        3. Condense context with a smaller LLM to reduce tokens
        4. Show condensed token count and compression ratio
        
        Args:
            question: User's question
            conversation_id: Optional conversation filter for memory retrieval
            limit: Max number of memory chunks to retrieve (default: backend-specific)
        
        Returns:
            Dict with 'raw' (original context + tokens) and 'condensed' (summarized + tokens)
        """
        memory_response = self._memory_query(
            question, conversation_id=conversation_id, limit=limit
        )

        items = memory_response.get("items") or []
        raw_context = memory_response.get("context")

        if not raw_context and items:
            formatted_chunks = [
                item.get("formatted")
                or item.get("text")
                or ""
                for item in items
            ]
            raw_context = "\n".join(chunk for chunk in formatted_chunks if chunk)

        raw_tokens = count_tokens(raw_context)

        condense_model = os.getenv("CONDENSE_MODEL", self.llm.chat_model)
        condense_max_tokens = int(os.getenv("CONDENSE_MAX_TOKENS", "400"))
        condense_temperature = float(os.getenv("CONDENSE_TEMPERATURE", "0"))

        if raw_context:
            summary_messages = [
                {
                    "role": "system",
                    "content": (
                        "You condense retrieved context into a concise, factual summary "
                        "for another model. Preserve key facts and speaker attributions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Context:\n{raw_context}\n\n"
                        "Produce a concise summary that keeps all actionable details."
                    ),
                },
            ]

            summary_result = self.llm.chat(
                summary_messages,
                model=condense_model,
                max_tokens=condense_max_tokens,
                temperature=condense_temperature,
            )
            condensed_text = summary_result.text.strip()
            summary_usage = summary_result.usage
        else:
            condensed_text = ""
            summary_usage = None

        condensed_tokens = count_tokens(condensed_text)
        compression_ratio = (
            condensed_tokens / raw_tokens if raw_tokens else None
        )

        return {
            "status": "success",
            "question": question,
            "memory": {
                "count": memory_response.get("count"),
                "conversation_id": memory_response.get("conversation_id"),
                "items": items,
            },
            "raw": {
                "tokens": raw_tokens,
                "characters": len(raw_context or ""),
                "context": raw_context,
            },
            "condensed": {
                "model": condense_model,
                "tokens": condensed_tokens,
                "characters": len(condensed_text),
                "summary": condensed_text,
                "usage": summary_usage,
                "compression_ratio": compression_ratio,
            },
        }

    def ask_question(
        self,
        question: str,
        conversation_id: Optional[str] = None,
        limit: Optional[int] = None,
        difficulty: Optional[str] = "auto",
    ) -> dict:
        """
        Full RAG pipeline: retrieve → condense → respond.
        
        1. Query memory for relevant context (optionally filtered by conversation_id)
        2. Condense retrieved context with a smaller LLM
        3. Send condensed context + question to responder for final answer
        
        Args:
            question: User's question
            conversation_id: Optional conversation filter
            limit: Max memory chunks to retrieve
            difficulty: (Reserved for future use)
        
        Returns:
            Dict with 'status', 'answer', 'context_used', and 'preview' (debug info)
        """
        try:
            preview = self.preview_context(
                question=question,
                conversation_id=conversation_id,
                limit=limit,
            )

            raw_ctx = (preview.get("raw") or {}).get("context") or ""
            items = (preview.get("memory") or {}).get("items") or []
            raw_tokens = count_tokens(raw_ctx)

            # 1) event inference (not hardcoded on Caroline)
            inferred_block = self._build_inferred_events_block(question, items)

            # 2) context policy
            if self._is_factoid(question) or raw_tokens < 900 or len(items) <= 8:
                final_context = inferred_block + raw_ctx
            else:
                light_ctx = self._build_light_context(question, items)
                final_context = inferred_block + light_ctx

            # 3) call responder
            response = requests.post(
                f"{self.responder_url}/respond",
                json={"question": question, "context": final_context},
                timeout=60,
            )
            response.raise_for_status()
            responder_result = response.json()
                
            if responder_result.get("status") != "success":
                    return {
                    "status": "error",
                    "error": responder_result.get("error", "Responder failed"),
                    "preview": preview,
                    }
            
            return {
                "status": "success",
                "answer": responder_result.get("answer"),
                "context_used": final_context,
                "preview": preview,
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }


coordinator = Coordinator(
    locomo_url=os.getenv("API_URL", "http://locomo:8000"),
    memory_url=os.getenv("MEMORY_URL", "http://memory:8005"),
    responder_url=os.getenv("RESPONDER_URL", "http://responder:8006"),
)


@app.get("/health")
async def health():
    try:
        requests.get(f"{coordinator.locomo_url}/health", timeout=5).raise_for_status()
        requests.get(f"{coordinator.memory_url}/health", timeout=5).raise_for_status()
        requests.get(f"{coordinator.responder_url}/health", timeout=5).raise_for_status()
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/load_conversations")
async def load_conversations(index: Optional[int] = None):
    try:
        if index is not None:
            return coordinator.load_conversation(index)
        else:
            return coordinator.load_all_conversations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/load_conversation/index/{index}")
# async def load_conversation_by_index(index: int):
#     try:
#         if index < 0 or index > 9:
#             raise HTTPException(status_code=400, detail="Index must be between 0 and 9")
#         return coordinator.load_conversation(index)
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/load_conversation/index/{index}")
async def start_load_conversation(index: int):
    if index < 0 or index > 9:
        raise HTTPException(status_code=400, detail="Index must be between 0 and 9")
    result = coordinator.start_load_conversation(index)
    return result


@app.get("/load_conversation/status/{job_id}")
async def load_conversation_status(job_id: str):
    result = coordinator.get_job_status(job_id)
    if result.get("status") == "error" and result.get("error") == "job not found":
        raise HTTPException(status_code=404, detail="Job not found")
    return result

@app.post("/preview")
async def preview(request: PreviewRequest):
    """
    Preview the context condensation pipeline without generating a final answer.
    
    Shows raw context from memory vs condensed summary, with token counts.
    Useful for debugging memory retrieval and evaluating compression efficiency.
    """
    try:
        return coordinator.preview_context(
            question=request.question,
            conversation_id=request.conversation_id,
            limit=request.limit,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Ask a question using the full RAG pipeline: retrieve → condense → respond.
    
    Returns the final answer along with debug info (preview) showing
    what context was retrieved and how it was condensed.
    """
    try:
        result = coordinator.ask_question(
            question=request.question,
            conversation_id=request.conversation_id,
            limit=request.limit,
            difficulty=request.difficulty,
        )
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)