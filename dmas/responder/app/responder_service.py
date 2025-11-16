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
            
            system_msg = """
                You are a helpful assistant. 
                You MUST answer questions using ONLY the information provided in the MEMORIES. 
                You ARE allowed to do simple reasoning and calculations using those MEMORIES 
                (for example, converting relative time expressions like 'last year' into a calendar year 
                if a dated event is available). 
                You MUST NOT use outside/world knowledge or invent facts that are not logically implied 
                by the MEMORIES. 
                If the MEMORIES do not contain enough information to answer, you MUST reply exactly: 
                "I don't know based on the given memories." 
                Be concise and factual.
            """

            user_msg = f"""
                MEMORIES:
                {memory}

                QUESTION:
                {question}

                Answer using ONLY the MEMORIES above.
                You may combine information from different memories and perform simple logical or temporal reasoning.
                For temporal questions, use timestamps from the memories and, if possible, convert relative expressions
                (e.g. 'last year', 'two years later') into human-readable dates based only on those timestamps.
                If the answer is not supported by the memories, reply exactly:
                "I don't know based on the given memories."
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