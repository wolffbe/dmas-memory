import os
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.models import ConversationStats
from app.locomo_service import LocomoService
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("locomo starting...")
    storage.load_from_json()
    if storage.is_loaded:
        logger.info(f"Ready with {len(storage.conversations)} conversations!")
    
    yield

app = FastAPI(title="locomo", version="1.0", lifespan=lifespan)

storage = LocomoService(
    data_url=os.getenv("DATA_URL", "https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json")
)

@app.get("/", tags=["Info"])
async def root():
    return {
        "service": "locomo",
        "version": "1.0",
        "description": "Loads and serves LOCOMO conversation data",
        "data_format": "locomo10.json format with sessions, qa, observations, summaries",
        "endpoints": {
            "GET /conversations": "Get all conversations (paginated)",
            "GET /conversations/{sample_id}": "Get specific conversation",
            "GET /conversations/index/{index}": "Get conversation by index",
            "GET /conversations/{sample_id}/sessions/{session_id}": "Get specific session by session_id",
            "GET /conversations/index/{conv_index}/sessions/{session_index}": "Get session by conversation and session index",
            "GET /conversations/{sample_id}/questions": "Get questions for conversation",
            "GET /conversations/index/{index}/questions": "Get conversation by index and questions",
            "GET /stats": "Get statistics",
            "GET /health": "Health check"
        }
    }


@app.get("/health", tags=["Info"])
async def health():
    return {
        "status": "healthy" if storage.is_loaded else "not_loaded",
        "conversations_loaded": storage.is_loaded,
        "total_conversations": len(storage.conversations),
        "total_sessions": len(storage.all_sessions),
        "total_questions": len(storage.all_questions)
    }


@app.get("/stats", response_model=ConversationStats, tags=["Info"])
async def get_stats():
    stats = storage.get_stats()
    return ConversationStats(**stats)


@app.get("/conversations", tags=["Conversations"])
async def get_conversations(skip: int = 0, limit: int = 10):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    limit = min(limit, 10)
    conversations = storage.get_all_conversations(skip, limit)
    
    return {
        "total": len(storage.conversations),
        "skip": skip,
        "limit": limit,
        "count": len(conversations),
        "conversations": [c.dict() for c in conversations]
    }


@app.get("/conversations/{sample_id}", tags=["Conversations"])
async def get_conversation(sample_id: str):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    conversation = storage.get_conversation(sample_id)
    
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation {sample_id} not found")
    
    return {
        "sample_id": conversation.sample_id,
        "speaker_a": conversation.speaker_a,
        "speaker_b": conversation.speaker_b,
        "sessions": conversation.sessions,
        "session_datetimes": conversation.session_datetimes
    }


@app.get("/conversations/index/{index}", tags=["Conversations"])
async def get_conversation_by_index(index: int):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    conversation = storage.get_conversation_by_index(index)
    
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation at index {index} not found")
    
    return {
        "sample_id": conversation.sample_id,
        "speaker_a": conversation.speaker_a,
        "speaker_b": conversation.speaker_b,
        "sessions": conversation.sessions,
        "session_datetimes": conversation.session_datetimes
    }


@app.get("/conversations/{sample_id}/sessions/{session_id}", tags=["Conversations"])
async def get_conversation_session_by_id(sample_id: str, session_id: str):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    session = storage.get_conversation_session_by_id(sample_id, session_id)
    
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found in conversation {sample_id}"
        )
    
    return session


@app.get("/conversations/index/{conv_index}/sessions/{session_index}", tags=["Conversations"])
async def get_conversation_session_by_index(conv_index: int, session_index: int):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    session = storage.get_conversation_session_by_conv_index(conv_index, session_index)
    
    if session is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_index} not found in conversation {conv_index}"
        )
    
    return session


@app.get("/conversations/{sample_id}/questions", tags=["Questions"])
async def get_questions_for_conversation(sample_id: str):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    questions = storage.get_questions(sample_id)
    
    return {
        "sample_id": sample_id,
        "total_questions": len(questions),
        "questions": questions
    }


@app.get("/conversations/index/{index}/questions", tags=["Questions"])
async def get_questions_by_conversation_index(index: int):
    if not storage.is_loaded:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    conversation = storage.get_conversation_by_index(index)
    
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation at index {index} not found")
    
    questions = storage.get_questions(conversation.sample_id)
    
    return {
        "conversation_index": index,
        "sample_id": conversation.sample_id,
        "total_questions": len(questions),
        "questions": questions
    }