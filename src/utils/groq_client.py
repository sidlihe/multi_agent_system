from langchain_groq import ChatGroq
import os
from pathlib import Path
import sys

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config.settings import settings
from src.modules.responses import LLMResponse, LLMUsage

#logging
from src.utils.logger import get_logger
logger = get_logger()
logger.info("Initialized Groq client with model: %s", settings.GROQ_MODEL)


GROQ_API_KEY = settings.GROQ_API_KEY
GROQ_MODEL = settings.GROQ_MODEL

def get_llm(temperature=0.7, json_mode=False):
    """Returns a configured ChatGroq instance."""
    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=temperature,
        **kwargs
    )

if __name__ == "__main__":
    
    llm = get_llm()
    choice = input("Do you have any question? (yes/no): ").strip().lower()
    if choice == "yes":
        question = input("Enter your question: ")
        response = llm.invoke(question)
    elif choice == "no":
        response = llm.invoke("What is 2 + 2?")
    else:
        logger.error("Invalid choice. Exiting.")
        sys.exit(0)

    usage_data = response.response_metadata.get("token_usage", {})
    model_name = response.response_metadata.get("model_name", GROQ_MODEL)

    # print("Raw LLM Response:", response)

    llm_response = LLMResponse(
        data=response.content,
        usage=LLMUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        ),
        model=model_name
    )
 
    logger.info("\n".join(f"{k}: {v}" for k, v in llm_response.model_dump().items()))