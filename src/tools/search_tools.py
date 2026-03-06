from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool



@tool
def web_search(query: str):
    """
    Search the web for up-to-date information.
    Useful for: Recent events, factual verification, coding documentation.
    """
    search = TavilySearchResults(max_results=3)
    try:
        results = search.invoke(query)
        return str(results)
    except Exception as e:
        return f"Search failed: {str(e)}"