# src/agents/supervisor.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import os, sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.utils.groq_client import get_llm
from src.config.settings import settings, AgentName
from src.utils.logger import get_logger
from src.graph.completion import CompletionStatus, create_initial_completion_state, update_completion_state
from src.modules.responses import Supervisor_RouteResponse

logger = get_logger(__name__)
logger.info("Initializing Supervisor agent module...")

def normalize_next_agent(next_agent: str) -> str:
    next_upper = (next_agent or "").upper().strip()
    if "RESEARCHER" in next_upper: return AgentName.RESEARCHER
    if "ANALYST" in next_upper:    return AgentName.ANALYST
    return AgentName.ANALYST

def supervisor_node(state):
    logger.info("Entering Supervisor Node.")
    
    recursion_depth = state.get("recursion_depth", 0)
    whiteboard = state.get("whiteboard", "")
    
    # Get or create completion state - THIS IS THE KEY
    completion_state: CompletionStatus = state.get("completion_state")
    if not completion_state:
        completion_state = create_initial_completion_state()
    
    # ───────────────────────────────────────────────
    # DETERMINISTIC RULES (Overrides LLM routing)
    # ───────────────────────────────────────────────
    
    # 1. REACH HARD LIMITS
    if recursion_depth >= 15:
        logger.warning(f"Max iterations ({recursion_depth}) reached. FINISH.")
        return {
            "next": "FINISH",
            "recursion_depth": recursion_depth + 1,
            "completion_state": completion_state
        }
    
    # 2. IF COMPLETION STATE SAYS WE'RE DONE → FINISH
    if completion_state.should_force_finish():
        logger.info(f"Completion achieved (confidence={completion_state.confidence:.2f}). FINISH.")
        return {
            "next": "FINISH",
            "recursion_depth": recursion_depth + 1,
            "completion_state": completion_state
        }
    
    # 3. IF EVALUATOR JUST RAN: Check result
    if completion_state.stage == "EVALUATION":
        if completion_state.is_complete:
            logger.info("Evaluation PASSED. Workflow complete.")
            return {
                "next": "FINISH",
                "recursion_depth": recursion_depth + 1,
                "completion_state": completion_state
            }
        elif completion_state.should_attempt_refinement():
            logger.info(f"Evaluation FAILED (attempt {completion_state.refinement_attempts}). Refining...")
            # Determine which agent to route to based on feedback
            feedback_lower = completion_state.last_evaluator_feedback.lower()
            if "research" in feedback_lower or "data" in feedback_lower or "information" in feedback_lower:
                logger.info("Routing back to RESEARCHER for more data.")
                updated_state = update_completion_state(completion_state, stage="RESEARCH")
                return {
                    "next": AgentName.RESEARCHER,
                    "recursion_depth": recursion_depth + 1,
                    "completion_state": updated_state
                }
            else:
                logger.info("Routing back to ANALYST for text refinement.")
                updated_state = update_completion_state(completion_state, stage="ANALYSIS")
                return {
                    "next": AgentName.ANALYST,
                    "recursion_depth": recursion_depth + 1,
                    "completion_state": updated_state
                }
        else:
            logger.warning("Max refinement attempts reached. FINISH with best effort.")
            return {
                "next": "FINISH",
                "recursion_depth": recursion_depth + 1,
                "completion_state": completion_state
            }
    
    # 4. IF ANALYSIS JUST FINISHED → Send to evaluation
    if completion_state.stage == "ANALYSIS" and "*** ANALYSIS COMPLETE ***" in whiteboard:
        logger.info("Analysis complete. Routing to EVALUATOR.")
        updated_state = update_completion_state(completion_state, stage="EVALUATION")
        return {
            "next": AgentName.EVALUATOR,
            "recursion_depth": recursion_depth + 1,
            "completion_state": updated_state
        }
    
    # ───────────────────────────────────────────────
    # LLM-BASED ROUTING (Only when above rules don't apply)
    # ───────────────────────────────────────────────
    system_prompt = f"""You are the Workflow Supervisor. Route intelligently based on what's needed next.
    Valid agents: {AgentName.RESEARCHER}, {AgentName.ANALYST}

    DECISION RULES:
    1. If user needs real-world data/facts NOT in whiteboard → {AgentName.RESEARCHER}
    2. If research data exists but needs analysis/summary → {AgentName.ANALYST}
    3. If evaluator feedback says "more research" → {AgentName.RESEARCHER}
    4. If evaluator feedback says "reformat/recalculate/clarify" → {AgentName.ANALYST}
    """

    user_request = "Unknown"
    if state.get("messages"):
        msg = state["messages"][0]
        user_request = msg.content if hasattr(msg, 'content') else str(msg)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Request: {req}\n\nWhiteboard:\n{wb}\n\nRoute to?")
    ])

    llm = get_llm(temperature=0.1)
    chain = prompt | llm.with_structured_output(Supervisor_RouteResponse)

    try:
        response = chain.invoke({"req": user_request, "wb": whiteboard})  # Truncate for context
        normalized = normalize_next_agent(response.next)
        logger.info(f"Routing to: {normalized} | Reason: {response.reasoning}")
        
        # Update stage based on routing
        new_stage = "RESEARCH" if normalized == AgentName.RESEARCHER else "ANALYSIS"
        updated_state = update_completion_state(completion_state, stage=new_stage)
        
        return {
            "next": normalized,
            "recursion_depth": recursion_depth + 1,
            "completion_state": updated_state
        }
    except Exception as e:
        logger.error(f"Supervisor routing error: {e}")
        return {
            "next": AgentName.ANALYST,
            "recursion_depth": recursion_depth + 1,
            "completion_state": completion_state
        }


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