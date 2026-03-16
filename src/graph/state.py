from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage,HumanMessage, AIMessage, SystemMessage
import operator
import os,sys
from pathlib import Path

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.logger import get_logger
from src.graph.completion import CompletionStatus, create_initial_completion_state

logger = get_logger()
logger.info("Initializing agent state module...")

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
    
    # The Shared Whiteboard (Appending with history)
    whiteboard: Annotated[str, merge_whiteboard]
    
    # Track iteration count to prevent infinite loops
    recursion_depth: int
    
    # NEW: Structured completion tracking (replaces string markers)
    completion_state: CompletionStatus

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    state: AgentState = {
        "messages": [],
        "next": "start",
        "whiteboard": "",
        "recursion_depth": 0
    }
    
    # Simulate adding messages
    state["messages"] += [HumanMessage(content="Hello")]
    state["messages"] += [HumanMessage(content="How are you?")]
    state["messages"] += [AIMessage(content="I am fine, thanks!")]

    # Simulate whiteboard updates
    state["whiteboard"] = merge_whiteboard(state["whiteboard"], "Initial thought: I need to research.")
    state["whiteboard"] = merge_whiteboard(state["whiteboard"], "After research: I found useful information.")
    
    logger.info(f"Final State:\n{state}")

    #cross check whiteboard merging logic
    logger.info(f"Whiteboard Content:\n{state['whiteboard']}")
