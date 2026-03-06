# Evaluator Agent Node
# src/agents/evaluator.py

import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.logger import get_logger
from src.utils.groq_client import get_llm
from src.config.settings import settings
from enum import Enum

logger = get_logger(__name__)
logger.info("Initializing Evaluator agent module...")

class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR

class EvaluationResult(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0 evaluating completeness and accuracy.")
    feedback: str = Field(description="Specific feedback on what is missing or incorrect.")
    is_passing: bool = Field(description="True if score >= 0.8, else False.")

def evaluator_node(state: dict) -> dict:
    """Evaluator node for quality assurance and completion verification."""
    logger.info("Entering Evaluator Node.")
    llm = get_llm(temperature=0.3).with_structured_output(EvaluationResult)
    
    # Extract user query safely
    user_query = "Unknown Request"
    if state.get("messages") and len(state["messages"]) > 0:
        first_msg = state["messages"][0]
        try:
            user_query = first_msg.content if hasattr(first_msg, 'content') else str(first_msg)
        except Exception:
            pass
    
    whiteboard = state.get("whiteboard", "")
    recursion_depth = state.get("recursion_depth", 0)
    
    # Create evaluation prompt
    prompt = (
        f"You are the Quality Control Evaluator.\n"
        f"User Request: {user_query}\n\n"
        f"Current Whiteboard State (Work done so far):\n{whiteboard}\n\n"
        "Evaluate if the current state fully and accurately addresses the user request. "
        "Score it 0.0 to 1.0 where 1.0 = perfectly complete, 0.0 = no progress. "
        "If score >= 0.75, the work is PASSED. Otherwise provide specific feedback for improvement."
    )
    
    try:
        logger.debug("Invoking Evaluator LLM...")
        eval_result = llm.invoke([("user", prompt)])
        
        logger.info(f"Evaluation Score: {eval_result.score:.2f} | Passing: {eval_result.is_passing}")
        logger.info(f"Feedback: {eval_result.feedback}")
        
        # Append evaluation to whiteboard
        eval_summary = f"\n\n[EVALUATOR] Score: {eval_result.score:.1f}/1.0 | Status: {'✓ PASS' if eval_result.is_passing else '✗ FAIL'}"
        if eval_result.feedback:
            eval_summary += f" | Feedback: {eval_result.feedback}"
        
        updated_whiteboard = whiteboard + eval_summary
        
        if eval_result.is_passing:
            logger.info("Evaluation PASSED. Workflow ready to conclude.")
            return {
                "messages": [HumanMessage(content="Evaluation complete - work passed quality check.")],
                "whiteboard": updated_whiteboard,
                "next": "FINISH"
            }
        else:
            # Failed evaluation: return to supervisor for refinement
            logger.info("Evaluation FAILED. Routing back to Supervisor for improvements.")
            return {
                "messages": [HumanMessage(content="Evaluation failed - requesting improvements.")],
                "whiteboard": updated_whiteboard,
                "next": AgentName.SUPERVISOR
            }
            
    except Exception as e:
        logger.error(f"Evaluator LLM execution failed: {e}", exc_info=True)
        error_msg = f"\n\n[EVALUATOR ERROR] Failed to evaluate: {str(e)[:150]}"
        return {
            "messages": [HumanMessage(content="Evaluation error - returning to Supervisor.")],
            "whiteboard": whiteboard + error_msg,
            "next": AgentName.SUPERVISOR
        }

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    logger.info("Testing Evaluator Node...")
    
    # FIX: Use HumanMessage object instead of raw dict
    mock_state_bad = {
        "messages":[HumanMessage(content="Write a 500 word report on AI in healthcare.")],
        "whiteboard": "Researcher: AI is good for healthcare."
    }
    
    try:
        result = evaluator_node(mock_state_bad)
        logger.info("=== FINAL NODE OUTPUT ===")
        logger.info(f"Next Node: {result['next']}")
        logger.info(f"Updated Whiteboard:\n{result['whiteboard']}")
    except Exception as e:
        logger.error(f"\n[NOTE] Execution failed: {e}")