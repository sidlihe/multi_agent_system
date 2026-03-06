from langgraph.graph import StateGraph, END

import os,sys
from pathlib import Path

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.graph.state import AgentState
from src.config.settings import settings
from src.agents.supervisor import supervisor_node
from src.agents.researcher import researcher_node
from src.graph.checkpoints import get_checkpointer
from src.agents.analyst import analyst_node
from src.agents.evaluator import evaluator_node

from enum import Enum
class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR

def create_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes - Use actual agent functions instead of stubs
    workflow.add_node(AgentName.SUPERVISOR, supervisor_node)
    workflow.add_node(AgentName.RESEARCHER, researcher_node)
    workflow.add_node(AgentName.ANALYST, analyst_node)
    workflow.add_node(AgentName.EVALUATOR, evaluator_node)

    # 2. Add Conditional Edges
    # Supervisor routes to appropriate agent based on 'next' field
    workflow.add_conditional_edges(
        AgentName.SUPERVISOR,
        lambda x: x.get("next", "FINISH"),
        {
            AgentName.RESEARCHER: AgentName.RESEARCHER,
            AgentName.ANALYST: AgentName.ANALYST,
            AgentName.EVALUATOR: AgentName.EVALUATOR,
            "FINISH": END
        }
    )

    # 3. Add Worker Edges - All workers report back to Supervisor
    workflow.add_edge(AgentName.RESEARCHER, AgentName.SUPERVISOR)
    workflow.add_edge(AgentName.ANALYST, AgentName.SUPERVISOR)
    workflow.add_edge(AgentName.EVALUATOR, AgentName.SUPERVISOR)

    # 4. Set Entry Point
    workflow.set_entry_point(AgentName.SUPERVISOR)

    # 5. Compile the graph
    # Note: Checkpointing is available but requires thread_id in config.
    # For simple stateless execution, compile without checkpointer.
    # Uncomment the line below to enable persistence:
    # return workflow.compile(checkpointer=get_checkpointer())
    
    return workflow.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from src.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Compiling Workflow Graph...")
    
    # Compile the graph
    app = create_graph()
    
    # ==========================================
    # 1. GENERATE MERMAID PNG (If not exists)
    # ==========================================
    image_path = os.path.join(project_root, "architecture_graph.png")
    
    if not os.path.exists(image_path):
        logger.info(f"Generating Mermaid PNG at: {image_path}")
        try:
            png_bytes = app.get_graph().draw_mermaid_png()
            with open(image_path, "wb") as f:
                f.write(png_bytes)
            logger.info(" Architecture PNG successfully created!")
        except Exception as e:
            logger.error(f"Failed to generate PNG (Internet connection required for Mermaid API): {e}")
    else:
        logger.info(f" Graph architecture PNG already exists at {image_path}. Skipping generation.")

    # ==========================================
    # 2. TEST THE WORKFLOW (Single Pass)
    # ==========================================
    logger.info("Starting Workflow Test Pass...")
    
    user_query = "What is the capital of France, and what is 500 * 2?"
    logger.info(f"User Query: {user_query}")
    
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "whiteboard": "",
        "recursion_depth": 0
    }
    
    try:
        # We use .stream() so we can see what each node outputs in real-time
        for step_output in app.stream(initial_state):
            for node_name, state_update in step_output.items():
                logger.info(f"\n{'='*50}")
                logger.info(f" NODE EXECUTED: {node_name}")
                logger.info(f"{'='*50}")
                
                # Print the routing decision
                if "next" in state_update:
                    logger.info(f" Routing to: {state_update['next']}")
                    
                # Print the whiteboard updates
                if "whiteboard" in state_update and state_update["whiteboard"]:
                    logger.info(f" Whiteboard Update:\n{state_update['whiteboard']}")
                    
    except Exception as e:
        logger.error(f"Workflow execution crashed: {e}", exc_info=True)