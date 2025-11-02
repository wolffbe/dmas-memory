import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dmas_llm.client import LLMClient


app = FastAPI(title="responder", version="1.0")


class RespondRequest(BaseModel):
    question: str
    context: str


class Responder:
    def __init__(self, model: str):
        if not os.getenv("MODEL"):
            os.environ["MODEL"] = model

        self.llm = LLMClient()
        # max_tokens: maximum number of tokens to generate in the completion
        self.max_tokens = int(os.getenv("MAX_TOKENS", "512"))
        # temperature: 0-2, lower is more deterministic (0 for factual responses)
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
    
    def respond(self, question: str, context: str) -> dict:
        try:
            if not context:
                context = "No relevant information available."
            
            prompt = f"""Based on the following context, answer the question.

                CONTEXT:
                {context}

                QUESTION: {question}

                Answer:"""
            
            result = self.llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            answer = result.text.strip()
            
            return {
                "status": "success",
                "answer": answer
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }


responder = Responder(
    model=os.getenv("MODEL", "gpt-4o-mini")
)


@app.get("/health")
async def health():
    try:
        if not responder.llm.use_api:
            raise RuntimeError("LLMClient is not configured for API usage")
        return {"status": "healthy", "model": responder.llm.chat_model}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/respond")
async def respond(request: RespondRequest):
    try:
        result = responder.respond(request.question, request.context)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)