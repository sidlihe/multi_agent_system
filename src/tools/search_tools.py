# from langchain_community.tools.tavily_search import TavilySearchResults
import os,sys
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from pathlib import Path
# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.utils.logger import get_logger
from src.modules.responses import SearchInput

logger = get_logger()

logger.info("Initializing search tools module...")

# Load environment variables from .env
load_dotenv()

@tool("web_search",args_schema=SearchInput)
def web_search(query: str):
    """
    Search the web for up-to-date information.
    Useful for: Recent events, factual verification, coding documentation.
    """
    # Explicitly pass the API key from environment
    search = TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))
    try:
        results = search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Search failed: {str(e)}"
    
if __name__ == "__main__":
    query = "What are the latest advancements in AI as of June 2024?"
    logger.info(web_search.run(SearchInput(query=query)))
