import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import uvicorn
from fastapi import FastAPI, HTTPException

import threading
import uuid

# job store in RAM
MEM_JOBS: Dict[str, Dict[str, Any]] = {}
MEM_JOBS_LOCK = threading.Lock()

# guard in RAM (optional)
INGESTED_KEYS = set()

MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "mem0").lower()
MEM0_DEFAULT_TOP_K = int(os.getenv("MEMORY_TOP_K", "6"))

app = FastAPI(title="memory", version="1.0")

if MEMORY_BACKEND == "graphiti":
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
    from graphiti_core.nodes import EpisodeType

    llm_config = LLMConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model=os.getenv("MODEL", "gpt-4o-mini"),
        small_model=os.getenv("SMALL_MODEL", os.getenv("MODEL", "gpt-4o-mini")),
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
    )

    embedding_config = OpenAIEmbedderConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        embedding_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        embedding_dim=int(os.getenv("EMBEDDING_DIMS", "1536")),
        base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
    )

    def build_graphiti_instance():
        client = OpenAIClient(config=llm_config)
        embedder = OpenAIEmbedder(config=embedding_config)
        reranker = OpenAIRerankerClient(client=client, config=llm_config)
        instance = Graphiti(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            os.getenv("NEO4J_USER", "neo4j"),
            os.getenv("NEO4J_PASSWORD", "password"),
            llm_client=client,
            embedder=embedder,
            cross_encoder=reranker,
        )
        return instance, client

    graphiti, graphiti_llm_client = build_graphiti_instance()
    print("✓ Initialized Graphiti memory backend")
else:
    from mem0 import Memory

    config: Dict[str, Any] = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": os.getenv("QDRANT_COLLECTION", "conversations"),
                "host": os.getenv("QDRANT_HOST", "localhost"),
                "port": int(os.getenv("QDRANT_PORT", "6333")),
                "embedding_model_dims": int(os.getenv("EMBEDDING_DIMS", "1536")),
            },
        },
        "llm": {
            "provider": "openai",
            "config": {
                "model": os.getenv("MEMORY_LLM_MODEL", os.getenv("MODEL", "gpt-4o-mini")),
                "temperature": float(os.getenv("MEMORY_LLM_TEMPERATURE", "0")),
                "max_tokens": int(os.getenv("MEMORY_LLM_MAX_TOKENS", "1200")),
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": os.getenv("EMBED_MODEL", "text-embedding-3-small"),
                "api_key": os.environ.get("OPENAI_API_KEY"),
            },
        },
    }

    try:
        from qdrant_client import QdrantClient
        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            from qdrant_client.http.models import Distance, VectorParams

        qdrant_host = config["vector_store"]["config"]["host"]
        qdrant_port = config["vector_store"]["config"]["port"]
        qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

        collections_to_check = ["conversations", "mem0migrations"]
        embedding_dims = config["vector_store"]["config"]["embedding_model_dims"]

        for collection_name in collections_to_check:
            try:
                qdrant_client.get_collection(collection_name)
                print(f"✓ Collection '{collection_name}' already exists")
            except Exception:
                try:
                    qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=embedding_dims, distance=Distance.COSINE),
                    )
                    print(f"✓ Created collection '{collection_name}'")
                except Exception as create_err:
                    if "already exists" in str(create_err).lower():
                        print(f"✓ Collection '{collection_name}' exists (race condition handled)")
                    else:
                        print(f"⚠️  Could not create collection '{collection_name}': {create_err}")

        m = Memory.from_config(config)
        print("✓ Initialized Mem0 memory backend")
    except Exception as e:
        print(f"❌ Failed to initialize memory backend: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    if MEMORY_BACKEND == "graphiti":
        print("Initializing Graphiti indices and constraints...")
        try:
            maybe_coro = graphiti.build_indices_and_constraints()
            if hasattr(maybe_coro, "__await__"):
                await maybe_coro
        except Exception as e:
            print(f"Warning: Could not initialize Graphiti indices: {e}")
            print("This is normal if indices already exist")


@app.on_event("shutdown")
async def shutdown_event():
    if MEMORY_BACKEND == "graphiti":
        print("Closing Graphiti connection...")
        await graphiti.close()
        print("✓ Graphiti connection closed")


@app.get("/health")
async def health():
    return {"status": "healthy", "backend": MEMORY_BACKEND}


@app.post("/memorize")
async def memorize(request: Dict[str, Any]):
    conv_index = request.get("conv_index")
    data = request.get("data")

    if conv_index is None or not data:
        raise HTTPException(status_code=400, detail="Missing required fields: conv_index, data")

    # se è graphiti teniamo il comportamento vecchio (di solito è più veloce)
    if MEMORY_BACKEND == "graphiti":
        return await memorize_graphiti(conv_index, data)

    # mem0: avviamo un job e torniamo SUBITO
    job_id = str(uuid.uuid4())
    with MEM_JOBS_LOCK:
        MEM_JOBS[job_id] = {
            "status": "pending",
            "conv_index": conv_index,
            "conversation_id": data.get("sample_id") or data.get("conversation_id"),
            "logs": [],
        }

    t = threading.Thread(
        target=run_mem0_job,
        args=(job_id, conv_index, data),
        daemon=True,
    )
    t.start()

    return {
        "status": "started",
        "job_id": job_id,
        "conv_index": conv_index,
    }

@app.get("/memorize/status/{job_id}")
async def memorize_status(job_id: str):
    with MEM_JOBS_LOCK:
        job = MEM_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job


async def memorize_mem0(conv_index: int, data: Dict[str, Any], logger=None):
    conversation_id = (
        data.get("sample_id")
        or data.get("conversation_id")
        or f"conversation-{conv_index}"
    )
    if logger:
        logger(f"[DEBUG] memorize_mem0: conv_index={conv_index}, conversation_id={conversation_id}")
    else:
        print(f"[DEBUG] memorize_mem0: conv_index={conv_index}, conversation_id={conversation_id}")

    sessions = data.get("sessions", {})
    session_datetimes = data.get("session_datetimes", {})

    memories_added = 0

    for session_key, turns in sessions.items():
        if not isinstance(turns, list):
            continue

        timestamp = session_datetimes.get(f"{session_key}_date_time")
        if logger:
            logger(f"[DEBUG] Processing {session_key}: {len(turns)} turns, timestamp={timestamp}")
        else:
            print(f"[DEBUG] Processing {session_key}: {len(turns)} turns, timestamp={timestamp}")

        # build the session document (list of lines)
        parts = []
        speakers = set()

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            text = (turn.get("text") or "").strip()
            if not text:
                continue

            speaker = (turn.get("speaker") or "unknown").strip()
            speakers.add(speaker)

            blip_caption = (turn.get("blip_caption") or "").strip()

            line = f"{speaker}: {text}"
            if blip_caption:
                line += f" [Image: {blip_caption}]"

            parts.append(line)

        if not parts:
            continue
        # build the session text (1 document = 1 string)
        session_text = "\n".join(parts)

        # split the session text into chunks depending on the length
        if len(session_text) <= 3500:
            chunks = [session_text]
        else:
            MAX_CHARS = 3500
            chunks = [session_text[i:i+MAX_CHARS] for i in range(0, len(session_text), MAX_CHARS)]

        for chunk_idx, chunk in enumerate(chunks):
            metadata = {
                "conversation_id": conversation_id,
                "conv_index": conv_index,
                "session": session_key,
                "timestamp": timestamp,
                "chunk_idx": chunk_idx,
                "speakers": list(speakers),
            }

             # --- RAM GUARD START ---
            key = f"{conversation_id}:{session_key}:{chunk_idx}"
            if key in INGESTED_KEYS:
                continue
            INGESTED_KEYS.add(key)
            # --- RAM GUARD END ---

            m.add(
                chunk,
                user_id=conversation_id,
                metadata=metadata,
            )
            memories_added += 1

    if logger:
        logger(f"[DEBUG] Total memories added (session-level): {memories_added}")
    else:
        print(f"[DEBUG] Total memories added (session-level): {memories_added}")
    return {
        "status": "success",
        "conversation_id": conversation_id,
        "memories_added": memories_added,
    }


def run_mem0_job(job_id: str, conv_index: int, data: Dict[str, Any]):
    def log(msg: str):
        with MEM_JOBS_LOCK:
            MEM_JOBS[job_id]["logs"].append(msg)

    with MEM_JOBS_LOCK:
        MEM_JOBS[job_id]["status"] = "running"

    try:
        # call the existing function
        import asyncio
        result = asyncio.run(memorize_mem0(conv_index, data, logger=log))

        with MEM_JOBS_LOCK:
            MEM_JOBS[job_id]["status"] = "done"
            MEM_JOBS[job_id]["result"] = result
        log("memorize finished")
    except Exception as e:
        with MEM_JOBS_LOCK:
            MEM_JOBS[job_id]["status"] = "error"
            MEM_JOBS[job_id]["error"] = str(e)
        log(f"error: {e}")

async def memorize_graphiti(conv_index: int, data: Dict[str, Any]):
    episodes_added = 0

    for key, value in data.items():
        if not key.startswith("session_") or key.endswith("_date_time"):
            continue

        turns = value
        if not isinstance(turns, list):
            continue

        timestamp_string = data.get(f"{key}_date_time")
        if not timestamp_string:
            raise HTTPException(status_code=400, detail=f"Missing timestamp for {key}")

        try:
            reference_time = datetime.strptime(timestamp_string, "%I:%M %p on %d %B, %Y").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            try:
                reference_time = datetime.strptime(timestamp_string, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid timestamp format for {key}: '{timestamp_string}'. "
                        "Expected '7:55 pm on 9 June, 2023' or 'YYYY-MM-DD HH:MM:SS'"
                    ),
                )

        episode_parts: List[str] = []
        speakers = set()

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            text = (turn.get("text") or "").strip()
            if not text:
                continue

            speaker = (turn.get("speaker") or "unknown").strip()
            blip_caption = (turn.get("blip_caption") or "").strip()
            speakers.add(speaker)

            line = f"{speaker}: {text}"
            if blip_caption:
                line += f" [Image: {blip_caption}]"
            episode_parts.append(line)

        if not episode_parts:
            continue

        episode_body = "\n".join(episode_parts)
        speaker_list = ", ".join(sorted(speakers))
        description = f"Conversation session with participants: {speaker_list} at {timestamp_string}"

        await graphiti.add_episode(
            name=f"Conversation {conv_index} - {key}",
            episode_body=episode_body,
            source=EpisodeType.message,
            source_description=description,
            reference_time=reference_time,
            group_id=data.get("conversation_id") or f"conversation-{conv_index}",
        )
        episodes_added += 1

    return {"status": "success", "episodes_added": episodes_added}


@app.post("/query")
async def query(request: Dict[str, Any]):
    prompt = request.get("prompt")
    conversation_id = request.get("conversation_id")
    limit = request.get("limit")

    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt field")

    if MEMORY_BACKEND == "graphiti":
        return await query_graphiti(prompt)
    else:
        return await query_mem0(prompt, conversation_id=conversation_id, limit=limit)


async def query_mem0(prompt: str, conversation_id: Optional[str] = None, limit: Optional[int] = None):
    try:
        top_k = MEM0_DEFAULT_TOP_K if not limit else max(1, int(limit))
    except ValueError:
        top_k = MEM0_DEFAULT_TOP_K

    search_kwargs: Dict[str, Any] = {"query": prompt, "limit": top_k}
    if conversation_id:
        search_kwargs["user_id"] = conversation_id

    search_results = m.search(**search_kwargs)
    print(f"[DEBUG] mem0.search returned: {type(search_results)} -> {search_results}")

    if isinstance(search_results, dict):
        memories = search_results.get("results", [])
    elif isinstance(search_results, list):
        memories = search_results
    else:
        memories = []

    items = []
    for result in memories:
        if isinstance(result, dict):
            memory_text = (result.get("memory") or "").strip()
            metadata = result.get("metadata") or {}
            score = result.get("score")

            session = metadata.get("session", "")
            timestamp = metadata.get("timestamp", "")
            speaker = metadata.get("speaker", "")

            header_bits = [bit for bit in (session, timestamp) if bit]
            header = f"[{' | '.join(header_bits)}] " if header_bits else ""
            speaker_prefix = f"{speaker}: " if speaker else ""
            formatted = f"{header}{speaker_prefix}{memory_text}".strip()

            items.append({"text": memory_text, "metadata": metadata, "formatted": formatted, "score": score})
        elif isinstance(result, str):
            items.append({"text": result, "metadata": {}, "formatted": result, "score": None})

    context_snippets = [item["formatted"] for item in items if item["formatted"]]
    context = "\n\n".join(context_snippets)

    response: Dict[str, Any] = {
        "status": "success",
        "items": items,
        "count": len(items),
        "conversation_id": conversation_id,
    }
    if context:
        response["context"] = context

    return response


async def query_graphiti(prompt: str):
    results = await graphiti.search(prompt)
    if results and len(results) > 0:
        center_node_uuid = results[0].source_node_uuid
        results = await graphiti.search(prompt, center_node_uuid=center_node_uuid)

    payload: Dict[str, Any] = {"status": "success"}

    if results:
        facts = [r.fact for r in results if getattr(r, "fact", None)]
        context = " ".join(facts)
        if context:
            payload["context"] = context

    return payload


@app.delete("/reset")
async def reset_memory():
    """Reset all memory (for testing purposes). WARNING: This clears ALL stored memories!"""
    global graphiti, graphiti_llm_client, m
    
    try:
        if MEMORY_BACKEND == "graphiti":
            driver = graphiti._driver
            with driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")

            await graphiti.close()
            graphiti, graphiti_llm_client = build_graphiti_instance()

            return {"status": "success", "message": "Graphiti memory reset (Neo4j cleared)", "backend": "graphiti"}
        else:
            m = Memory.from_config(config)
            return {
                "status": "success",
                "message": "Mem0 memory reset (reinitialized)",
                "backend": "mem0",
                "note": "Qdrant collection still exists but Memory instance is fresh",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/stats")
async def get_memory_stats():
    try:
        if MEMORY_BACKEND == "graphiti":
            return {
                "status": "success",
                "backend": "graphiti",
                "note": "Stats not yet implemented for Graphiti",
                "nodes": "unknown",
                "edges": "unknown",
            }
        else:
            try:
                _ = m.search(query="test", user_id="system", limit=1)
                ok = True
                note = "Memory is operational. Use Qdrant API directly for detailed stats."
            except Exception as e:
                ok = False
                note = f"Memory operational but test query failed: {e}"

            return {
                "status": "success",
                "backend": "mem0",
                "collection": config["vector_store"]["config"]["collection_name"],
                "qdrant_host": config["vector_store"]["config"]["host"],
                "qdrant_port": config["vector_store"]["config"]["port"],
                "note": note,
                "test_query_success": ok,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
