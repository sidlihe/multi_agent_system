# src/agents/evaluator.py

import os
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.logger import get_logger
from src.utils.groq_client import get_llm
from src.config.settings import settings
from src.graph.completion import CompletionStatus, update_completion_state
from enum import Enum
from src.modules.responses import EvaluationResult

logger = get_logger(__name__)
logger.info("Initializing Evaluator agent module...")

class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR



def evaluator_node(state: dict) -> dict:
    """
    Evaluator node: Quality assurance with structured completion tracking.
    CRITICAL: Uses completion_state to avoid loops, not string markers.
    """
    logger.info("Entering Evaluator Node.")
    
    # Get current completion state
    completion_state: CompletionStatus = state.get("completion_state")
    if not completion_state:
        from src.graph.completion import create_initial_completion_state
        completion_state = create_initial_completion_state()
    
    # Extract user query
    user_query = "Unknown Request"
    if state.get("messages") and len(state["messages"]) > 0:
        first_msg = state["messages"][0]
        try:
            user_query = first_msg.content if hasattr(first_msg, 'content') else str(first_msg)
        except Exception:
            pass
    
    whiteboard = state.get("whiteboard", "")
    recursion_depth = state.get("recursion_depth", 0)
    
    # Evaluation prompt - CLEAR THRESHOLD
    prompt = (
        f"You are the Quality Control Evaluator.\n"
        f"User Request: {user_query}\n\n"
        f"Current Answer in Whiteboard:\n{whiteboard}\n\n"
        "EVALUATION RULES:\n"
        "1. Score 0.0-1.0 where 0.75+ = sufficient for user's request.\n"
        "2. is_passing should be True ONLY if BOTH score >= 0.75 AND the answer fully addresses the request.\n"
        "3. is_passing = False if ANY key part of the request is missing or incomplete.\n"
        "4. For multi-part questions (e.g., 'when, who, what'), ensure ALL parts are covered.\n"
        "5. Be strict: missing one part = failure, even if score is 0.80.\n"
    )
    
    try:
        # Use low temperature for consistent evaluation
        llm = get_llm(temperature=0.1).with_structured_output(EvaluationResult)
        logger.debug("Evaluator scoring...")
        eval_result = llm.invoke([("user", prompt)])
        
        # CRITICAL FIX: Honor the LLM's is_passing decision, not just the score
        is_passing = eval_result.is_passing
        
        logger.info(f"Score: {eval_result.score:.2f} | Status: {'PASS' if is_passing else 'FAIL'}")
        
        # Update completion state based on evaluation
        new_completion_state = update_completion_state(
            completion_state,
            stage="EVALUATION",
            confidence=eval_result.score,
            is_complete=is_passing,
            feedback=eval_result.feedback if not is_passing else ""
        )
        
        # Add evaluation summary to whiteboard
        eval_summary = f"\n\n[EVALUATOR #{recursion_depth}] Score: {eval_result.score:.2f}/1.0 | {'✓ PASS' if is_passing else '✗ FAIL'}"
        if not is_passing and eval_result.feedback:
            eval_summary += f"\nFeedback: {eval_result.feedback}"
            eval_summary += f"\nHint: {new_completion_state.get_refinement_hint()}"
        
        updated_whiteboard = whiteboard + eval_summary
        logger.info(f"Evaluation summary added to whiteboard: {eval_summary}")
        logger.info("Evaluation complete.")
        
        return {
            "messages": [AIMessage(content=f"Eval: {'PASS' if is_passing else 'FAIL'}")],
            "whiteboard": eval_summary,  # Return only the new update
            "next": AgentName.SUPERVISOR,
            "completion_state": new_completion_state,
            "recursion_depth": recursion_depth + 1
        }
            
    except Exception as e:
        logger.error(f"Evaluator failed: {e}")
        error_msg = f"\n\n[EVALUATOR ERROR] {str(e)[:100]}"
        # Mark as failed but continue
        new_completion_state = update_completion_state(
            completion_state,
            stage="EVALUATION",
            confidence=0.0,
            is_complete=False,
            feedback=f"Evaluator error: {str(e)[:50]}"
        )
        return {
            "messages": [AIMessage(content="Evaluation error.")],
            "whiteboard": error_msg,
            "next": AgentName.SUPERVISOR,
            "completion_state": new_completion_state,
            "recursion_depth": recursion_depth + 1
        }

if __name__ == "__main__":
    logger.info("Testing Evaluator Node...")
    mock_state = {
        "messages": [HumanMessage(content="Should I buy IRFC stock now?")],
        "whiteboard": """Researcher searched: IRFC stock price today
        Result: current price ~99.5 INR

        Analyst Insights:
        Current price: 99.5 INR
        Recent trend: down 30% from 52w high
        *** ANALYSIS COMPLETE ***""",
                "recursion_depth": 2
    }
    
    try:
        result = evaluator_node(mock_state)
    #     print("\n" + "="*60)
    #     print("EVALUATOR NODE OUTPUT")
    #     print("="*60)
    #     print(f"Next: {result.get('next')}")
    #     print(f"Whiteboard update length: {len(result.get('whiteboard',''))}")
    #     print("Last part of whiteboard:")
    #     print(result['whiteboard'][-400:])
    except Exception as e:
        logger.error(f"Test failed: {e}")