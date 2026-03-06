# src/agents/supervisor.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os,sys
from pathlib import Path

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.groq_client import get_llm
from src.config.settings import settings
from src.utils.logger import get_logger
# Initialize logger
logger = get_logger()
logger.info("Initializing Supervisor agent module...")

#enum for agent names
from enum import Enum
class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR

# 1. Define the Schema for Routing
class RouteResponse(BaseModel):
    next: str = Field(
        description=(
            "The next agent to act. Must be one of: "
            "'Researcher' (for web search), 'Analyst' (for analysis), "
            "'Evaluator' (for quality check), or 'FINISH' to end the workflow."
        )
    )
    reasoning: str = Field(description="Brief reason for the selection.")

def normalize_next_agent(next_agent: str) -> str:
    """Normalize agent names to match expected values."""
    if not next_agent:
        return "FINISH"
    
    # Normalize case-insensitive matching
    next_upper = next_agent.upper().strip()
    
    if "RESEARCHER" in next_upper:
        return AgentName.RESEARCHER
    elif "ANALYST" in next_upper:
        return AgentName.ANALYST
    elif "EVALUATOR" in next_upper:
        return AgentName.EVALUATOR
    elif "FINISH" in next_upper:
        return "FINISH"
    else:
        logger.warning(f"Unknown agent name '{next_agent}', defaulting to FINISH")
        return "FINISH"

# 2. System Prompt
system_prompt = (
    f"You are a Supervisor managing a team:\n"
    f"  - {AgentName.RESEARCHER} (web search)\n"
    f"  - {AgentName.ANALYST} (logic/data analysis)\n"
    f"  - {AgentName.EVALUATOR} (quality control)\n\n"
    f"IMPORTANT: When routing, use EXACT names from the list above.\n"
    f"Valid next values: '{AgentName.RESEARCHER}', '{AgentName.ANALYST}', '{AgentName.EVALUATOR}', 'FINISH'\n\n"
    f"1. Review the user request and conversation history.\n"
    f"2. Check the 'Whiteboard' (shared state) for intermediate progress.\n"
    f"3. Decide who acts next. If the task is complete, select FINISH."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Current Whiteboard State:\n{whiteboard}\n\nSelect the next worker:")
])

def supervisor_node(state):
    """Supervisor node for routing decisions and orchestration."""
    logger.info("Entering Supervisor Node.")
    
    recursion_depth = state.get("recursion_depth", 0)
    logger.debug(f"Recursion depth: {recursion_depth}")
    
    llm = get_llm(temperature=0.3)
    # Bind the schema to force structured output
    chain = prompt | llm.with_structured_output(RouteResponse)
    
    try:
        response = chain.invoke(state)
        
        # Normalize the agent name to handle case sensitivity
        normalized_next = normalize_next_agent(response.next)
        logger.info(f"Supervisor Decision: {response.next} -> {normalized_next} | Reasoning: {response.reasoning}")
        
        return {
            "next": normalized_next,
            "recursion_depth": recursion_depth + 1
        }
    except Exception as e:
        logger.error(f"Supervisor decision failed: {e}")
        logger.info("Defaulting to FINISH due to error.")
        # Fail-safe: finish if supervisor fails to prevent infinite loops
        return {
            "next": "FINISH",
            "recursion_depth": recursion_depth + 1
        }

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    print("Starting direct execution test for Supervisor Node...")
    
    mock_state = {
        "messages":[("human", "What is the capital of France?")],
        "whiteboard": "No information yet.",
        "recursion_depth": 0
    }
    
    result = supervisor_node(mock_state)
    print(f"Supervisor Node Output: {result}")