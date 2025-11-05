from fastapi import FastAPI, HTTPException
import os

from app.coordinator_service import CoordinatorService

app = FastAPI(title="coordinator", version="1.0")

coordinator = CoordinatorService(
    locomo_url=os.getenv("LOCOMO_URL"),
    memory_url=os.getenv("MEMORY_URL"),
    responder_url=os.getenv("RESPONDER_URL"),
    ollama_model=os.getenv("OLLAMA_MODEL")
)

@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }

@app.post("/ask")
async def ask(question: str):
    try:
        result = coordinator.ask(
            question=question
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/conversation/load/{index}")
async def load_conversation(index: int):
    if index < 0 or index > 9:
        raise HTTPException(status_code=400, detail="Index must be between 0-9")
    
    result = coordinator.load_conversation(index)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result

@app.post("/conversation/load/{conversation_id}/session/{session_id}")
async def load_conversation(conversation_id: int, session_id: int):
    result = coordinator.load_conversation_session(conversation_id, session_id)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("error"))
    
    return result