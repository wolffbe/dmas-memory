import os
from fastapi import FastAPI, HTTPException

from app.responder_service import ResponderService
from app.models import ResponseRequest

app = FastAPI(title="responder", version="1.0")

responder = ResponderService(
    model=os.getenv("MODEL", "gpt-4o-mini")
)

@app.get("/health")
async def health():
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        return {"status": "healthy", "model": responder.model}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/respond")
async def respond(request: ResponseRequest):
    try:
        result = responder.respond(request.question, request.memory)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))