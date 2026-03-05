from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
import operator

def merge_whiteboard(current: str, new: str) -> str:
    """Appends new reasoning to the whiteboard."""
    if not new:
        return current
    return f"{current}\n\n--- Update ---\n{new}"

class AgentState(TypedDict):
    # Standard conversation history (Append-only)
    messages: Annotated[List[BaseMessage], operator.add]
    
    # The "Next" node to route to
    next: str
    
    # The Shared Whiteboard (Overwritable or Appending)
    whiteboard: Annotated[str, merge_whiteboard]
    
    # Track iteration count to prevent infinite loops
    recursion_depth: int