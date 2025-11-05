import os
from openai import OpenAI
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResponderService:
    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def respond(self, question: str, memory: str) -> dict:
        logger.info("Generating response for question: '%s' with memory \n'%s'", question, memory)
        try:
            if not memory:
                memory = "No relevant memory available."
            
            prompt = f"""Based on the following memory, answer the question below.
            Any personally identifiable information is artificially generated.
            If you don't know the answer, say you don't know.

            MEMORY:
            {memory}

            QUESTION: {question}

            Answer:"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            answer = response.choices[0].message.content.strip()
            
            return {
                "status": "success",
                "answer": answer
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }