# src/agents/supervisor.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
import os, sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.groq_client import get_llm
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger()
logger.info("Initializing Supervisor agent module...")

from enum import Enum
class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR

class RouteResponse(BaseModel):
    next: str = Field(description="The next agent to act.")
    reasoning: str = Field(description="Brief reason for the selection.")

def normalize_next_agent(next_agent: str) -> str:
    next_upper = (next_agent or "").upper().strip()
    if "RESEARCHER" in next_upper: return AgentName.RESEARCHER
    if "ANALYST" in next_upper:    return AgentName.ANALYST
    if "EVALUATOR" in next_upper:  return AgentName.EVALUATOR
    if "FINISH" in next_upper:     return "FINISH"
    return "FINISH"

system_prompt = f"""You are the Supervisor.
Valid next agents: '{AgentName.RESEARCHER}', '{AgentName.ANALYST}', '{AgentName.EVALUATOR}', 'FINISH'

ROUTING RULES:
1. If whiteboard contains '*** ANALYSIS COMPLETE ***' → route to EVALUATOR
2. If recent research data exists and no analysis yet → ANALYST
3. If task looks complete → FINISH
4. IMPORTANT SAFETY: If Evaluator has failed 2 or more times or recursion_depth >= 6 → FORCE FINISH immediately.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Current Whiteboard:\n{whiteboard}\n\nDecide next step:")
])

def supervisor_node(state):
    logger.info("Entering Supervisor Node.")
    recursion_depth = state.get("recursion_depth", 0)
    whiteboard = state.get("whiteboard", "")

    # ───────────────────────────────────────────────
    # NEW: STRONG LOOP PREVENTION
    # ───────────────────────────────────────────────
    evaluator_fail_count = whiteboard.count("[EVALUATOR") + whiteboard.count("✗ FAIL")
    
    if recursion_depth >= 6 or evaluator_fail_count >= 2:
        logger.info(f"SAFETY TRIGGER: Recursion={recursion_depth} | Evaluator fails={evaluator_fail_count}. Forcing FINISH to prevent infinite loop.")
        return {
            "next": "FINISH",
            "recursion_depth": recursion_depth + 1
        }

    # Original logic (kept for normal flow)
    if "*** ANALYSIS COMPLETE ***" in whiteboard:
        logger.info("Analysis completion signal detected → Routing to Evaluator")
        return {"next": AgentName.EVALUATOR, "recursion_depth": recursion_depth + 1}

    llm = get_llm(temperature=0.3)
    chain = prompt | llm.with_structured_output(RouteResponse)

    try:
        response = chain.invoke(state)
        normalized = normalize_next_agent(response.next)
        logger.info(f"Supervisor Decision: {response.next} → {normalized} | Reason: {response.reasoning}")
        return {"next": normalized, "recursion_depth": recursion_depth + 1}
    except Exception as e:
        logger.error(f"Supervisor failed: {e}")
        return {"next": "FINISH", "recursion_depth": recursion_depth + 1}

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    logger.info("Starting direct execution test for Supervisor Node...")
    
    mock_state = {
        "messages":[("human", "What is the capital of France?")],
        "whiteboard": "No information yet.",
        "recursion_depth": 0
    }
    
    result = supervisor_node(mock_state)
    logger.info(f"Supervisor Node Output: {result}")