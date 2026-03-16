# src/main.py
import sys
import os
import uuid
from pathlib import Path
from langchain_core.messages import HumanMessage

# Force UTF-8 output on Windows to prevent UnicodeEncodeError
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from src.utils.logger import get_logger
from src.utils.tracing import init_tracing
from src.config.settings import settings,AgentName
from src.graph.workflow import create_graph

logger = get_logger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("Starting Multi-Agent System (Groq + LangGraph)")
    logger.info("=" * 70)

    # Initialize tracing
    tracing_client = init_tracing()
    if tracing_client:
        logger.info("[OK] LangSmith tracing is ACTIVE")
    else:
        logger.warning("[WARN] LangSmith tracing not initialized")

    # Compile the graph
    try:
        app = create_graph()
        logger.info("[OK] Workflow graph compiled successfully")
    except Exception as e:
        logger.error(f"Failed to compile graph: {e}", exc_info=True)
        return

    # Get user input
    user_input = input("\nUser: ").strip()
    if not user_input:
        logger.warning("No input provided. Exiting.")
        print("No question entered. Goodbye.")
        return

    logger.info(f"Query: {user_input}")
    logger.info("=" * 70)

    # Prepare initial state with completion tracking
    from src.graph.completion import create_initial_completion_state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "whiteboard": "",
        "next": AgentName.SUPERVISOR,
        "recursion_depth": 0,
        "completion_state": create_initial_completion_state()
    }

    thread_id = str(uuid.uuid4())

    try:
        final_whiteboard = None
        last_node = None

        for event in app.stream(
            initial_state,
            config={
                "recursion_limit": settings.MAX_ITERATIONS + 5,  # Buffer for safe termination
                "configurable": {"thread_id": thread_id}
            }
        ):
            for node_name, node_state in event.items():
                if node_name != last_node:
                    logger.info(f"[Active Node: {node_name}]")
                    last_node = node_name
                    
                    if "next" in node_state and node_name == AgentName.SUPERVISOR:
                        logger.info(f"  → Routing Next to: {node_state['next']}")

                if "whiteboard" in node_state:
                    final_whiteboard = node_state["whiteboard"]

        # ───────────────────────────────────────────────
        # Show final result to the user
        # ───────────────────────────────────────────────
        print("\n" + "=" * 80)
        print(" FINAL ANSWER ".center(80, "="))
        print("=" * 80)

        if final_whiteboard and final_whiteboard.strip():
            print(final_whiteboard.strip())
        else:
            print("No final answer was generated.")
            print("(Check logs for details or increase recursion limit)")

        print("=" * 80)

    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"Workflow crashed: {e}", exc_info=True)
        logger.error("=" * 70)
        print("\n" + "=" * 80)
        print(" ERROR ".center(80, "="))
        print("The system encountered an error. See logs for details.")
        print("=" * 80)

if __name__ == "__main__":
    main()