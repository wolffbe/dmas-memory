from pydantic import BaseModel

class ResponseRequest(BaseModel):
    question: str
    memory: str