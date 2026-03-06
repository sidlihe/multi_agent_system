# Tracing setup for LangSmith observability in the multi-agent system.
# src/utils/tracing.py

import os,sys
from langsmith import Client
from dotenv import load_dotenv
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.logger import get_logger

logger = get_logger()
logger.info("Initializing tracing module...")

def init_tracing() -> Client | None:
    """
    Validates LangSmith tracing environment variables and initializes the client.
    Call this early in main.py to ensure observability is active.
    """
    load_dotenv()
    
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    project = os.environ.get("LANGCHAIN_PROJECT", "default")
    
    if tracing_enabled and api_key:
        try:
            client = Client()
            logger.info(f"LangSmith tracing ENABLED. Project: '{project}'")
            return client
        except Exception as e:
            logger.warning(f"LangSmith client failed to initialize: {e}")
            return None
    else:
        logger.info("LangSmith tracing is DISABLED. "
                    "Set LANGCHAIN_TRACING_V2=True and LANGCHAIN_API_KEY in .env to enable.")
        return None

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    logger.info("Testing LangSmith Tracing Setup...")
    
    # Force load environment variables for testing
    load_dotenv()
    
    client = init_tracing()
    
    if client:
        logger.info(" Successfully connected to LangSmith!")
        try:
            # Test API connection by listing projects
            projects = list(client.list_projects())
            project_names = [p.name for p in projects]
            logger.info(f" Found {len(projects)} projects in your account.")
            if os.environ.get("LANGCHAIN_PROJECT") in project_names:
                logger.info(f" Project '{os.environ.get('LANGCHAIN_PROJECT')}' is ready to receive traces.")
        except Exception as e:
            logger.error(f" Could not fetch projects. Check your API key. Error: {e}")
    else:
        logger.info("\n Tracing is not configured properly.")
        logger.info("Please add to your .env:")
        logger.info("LANGCHAIN_TRACING_V2=True")
        logger.info("LANGCHAIN_API_KEY=lsv2_pt_...")
        logger.info("LANGCHAIN_PROJECT=Multi-Agent-System")