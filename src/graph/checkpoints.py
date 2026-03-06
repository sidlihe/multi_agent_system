# Checkpointing and Rollback Logic for Supervisor Node
# src/graph/checkpoints.py

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

import os,sys
from pathlib import Path
# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.logger import get_logger
logger = get_logger()
logger.info("Initializing Checkpointing module...")

def get_checkpointer() -> MemorySaver:
    """Returns an in-memory saver for development. Use PostgresSaver for production."""
    return MemorySaver()

def rollback_to_last_supervisor(graph: CompiledStateGraph, config: dict):
    """
    Inspects the history of a thread, finds the last time the Supervisor
    was successfully executed, and rolls the graph state back to that point.
    """
    history = list(graph.get_state_history(config))
    
    if not history:
        logger.info("No history found to roll back.")
        return False
        
    for state_snapshot in history:
        # Check if the node that generated this state was the supervisor
        if state_snapshot.metadata and state_snapshot.metadata.get("source") == "Supervisor":
            logger.info(f"Rolling back to checkpoint: {state_snapshot.config['configurable']['checkpoint_id']}")
            # Update graph state to this snapshot
            graph.update_state(config, state_snapshot.values)
            return True
            
    logger.info("No stable Supervisor state found in history.")
    return False

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    logger.info("Testing Checkpoint & Rollback logic...")
    from langgraph.graph import StateGraph, START
    from typing import TypedDict
    
    class DummyState(TypedDict):
        val: int
        
    def add_one(state): return {"val": state["val"] + 1}
    def supervisor_mock(state): return {"val": state["val"]}
    
    builder = StateGraph(DummyState)
    builder.add_node("Supervisor", supervisor_mock)
    builder.add_node("Worker", add_one)
    builder.add_edge(START, "Supervisor")
    builder.add_edge("Supervisor", "Worker")
    
    checkpointer = get_checkpointer()
    graph = builder.compile(checkpointer=checkpointer)
    
    config = {"configurable": {"thread_id": "test_thread_1"}}
    
    # Run the graph
    graph.invoke({"val": 0}, config)
    current_state = graph.get_state(config)
    logger.info(f"State after execution: {current_state.values}")
    
    # Test Rollback
    rollback_to_last_supervisor(graph, config)
    rolled_back_state = graph.get_state(config)
    logger.info(f"State after rollback: {rolled_back_state.values}")