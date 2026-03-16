from langgraph.graph import StateGraph, END

import os
import sys
import base64
import zlib
import urllib.request
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

    # 1. Add Nodes
    workflow.add_node(AgentName.SUPERVISOR, supervisor_node)
    workflow.add_node(AgentName.RESEARCHER, researcher_node)
    workflow.add_node(AgentName.ANALYST, analyst_node)
    workflow.add_node(AgentName.EVALUATOR, evaluator_node)

    # 2. Add Conditional Edges
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

    # 3. Add Worker Edges
    workflow.add_edge(AgentName.RESEARCHER, AgentName.SUPERVISOR)
    workflow.add_edge(AgentName.ANALYST, AgentName.SUPERVISOR)
    workflow.add_edge(AgentName.EVALUATOR, AgentName.SUPERVISOR)

    # 4. Set Entry Point
    workflow.set_entry_point(AgentName.SUPERVISOR)

    return workflow.compile()

def generate_robust_mermaid_png(app, output_path: str, logger):
    """
    Bulletproof function to generate Mermaid PNGs using HTTP POST to Kroki.io.
    Sending data via POST body prevents URL length limits entirely.
    """
    import urllib.request
    
    try:
        mermaid_text = app.get_graph().draw_mermaid()
        logger.info("Connecting to Kroki.io image server via POST request...")
        
        url = "https://kroki.io/mermaid/png"
        data = mermaid_text.encode('utf-8')
        
        # Using POST by providing the 'data' parameter
        req = urllib.request.Request(
            url, 
            data=data, 
            headers={'Content-Type': 'text/plain', 'User-Agent': 'Mozilla/5.0'}
        )
        
        with urllib.request.urlopen(req) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
                
        logger.info(f"✅ Architecture PNG successfully created at: {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to generate PNG via API: {e}")
        # Ultimate fallback: Save as text so you have SOMETHING to look at
        fallback_path = output_path.replace(".png", ".md")
        try:
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write("```mermaid\n" + app.get_graph().draw_mermaid() + "\n```")
            logger.info(f"⚠️ Saved raw Markdown diagram instead at: {fallback_path}")
        except:
            pass

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from src.utils.logger import get_logger
    
    logger = get_logger(__name__)
    logger.info("Compiling Workflow Graph...")
    
    # Compile the graph
    app = create_graph()
    
    # ==========================================
    # 1. GENERATE MERMAID PNG (Robust Method)
    # ==========================================
    image_path = os.path.join(project_root, "architecture_graph.png")
    generate_robust_mermaid_png(app, image_path, logger)

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