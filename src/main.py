import sys, os
from langchain_core.messages import HumanMessage
from pathlib import Path
import uuid

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from src.utils.logger import get_logger
from src.utils.tracing import init_tracing
from src.config.settings import settings

logger = get_logger()

from src.graph.workflow import create_graph

def main():
    logger.info("="*60)
    logger.info("Initializing Multi-Agent System (Groq + LangGraph)...")
    logger.info("="*60)
    
    # Initialize and verify LangSmith tracing
    tracing_client = init_tracing()
    if tracing_client:
        logger.info("[OK] LangSmith tracing is ACTIVE")
    
    # Create and compile the workflow graph
    app = create_graph()
    logger.info("[OK] Workflow graph compiled successfully")
    
    # Get user input
    user_input = input("\nUser: ").strip()
    if not user_input:
        logger.warning("No input provided.")
        return
    
    # Initialize state with recursion depth tracking
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "whiteboard": "",
        "next": settings.SUPERVISOR,
        "recursion_depth": 0
    }

    logger.info(f"Starting workflow with input: {user_input[:50]}...")
    logger.info("="*60)

    # Run the Graph with recursion limit and thread ID for checkpointing
    try:
        # Generate a unique thread ID for this conversation
        thread_id = str(uuid.uuid4())
        
        for event in app.stream(
            initial_state,
            config={
                "recursion_limit": settings.MAX_ITERATIONS,
                "configurable": {"thread_id": thread_id}
            }
        ):
            for node_name, node_state in event.items():
                logger.info(f"\n[Node: {node_name}]")
                
                # Log whiteboard updates
                if "whiteboard" in node_state:
                    wb = node_state["whiteboard"]
                    if len(wb) > 500:
                        logger.info(f"[Whiteboard]: {wb[:500]}...\n[Truncated]")
                    else:
                        logger.info(f"[Whiteboard]: {wb}")
                
                # Log next routing decision
                if "next" in node_state:
                    logger.info(f"[Next Route]: -> {node_state['next']}")
        
        logger.info("="*60)
        logger.info("Workflow completed successfully.")
        logger.info(f"Traces available at: https://smith.langchain.com/")
                    
    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"System Error: {e}", exc_info=True)
        logger.error("="*60)
        raise

if __name__ == "__main__":
    main()