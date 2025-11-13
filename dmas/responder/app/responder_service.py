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
            
            system_msg = (
                "You are a helpful assistant that MUST answer questions "
                "using ONLY the information contained in the provided memories. "
                "If the memories do not contain the answer, you MUST say that you don't know. "
                "Do NOT add extra facts or assumptions. "
                "Be concise but complete."
            )

            user_msg = f"""MEMORIES:
                {memory}

                QUESTION: {question}

                Answer based only on the MEMORIES above.
                If the answer cannot be found in the memories, say explicitly: "I don't know based on the given memories."
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
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