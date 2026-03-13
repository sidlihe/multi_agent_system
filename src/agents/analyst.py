# src/agents/analyst.py
from langchain_core.messages import HumanMessage, AIMessage

import os, sys
from pathlib import Path

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.groq_client import get_llm
from src.tools.data_tools import calculate_math, profile_dataframe
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Initializing Analyst agent module...")

from enum import Enum
class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR

ANALYST_TOOLS =[calculate_math, profile_dataframe]

def analyst_node(state: dict) -> dict:
    logger.info("Entering Analyst Node.")
    llm = get_llm(temperature=0.3)
    llm_with_tools = llm.bind_tools(ANALYST_TOOLS)
    
    system_prompt = (
        "You are the Data Analyst. Your job is to analyze quantitative data.\n"
        "1. Read the Whiteboard for context.\n"
        "2. Use math or data tools if calculations are needed.\n"
        "3. Provide a clear summary with breakdown.\n"
        "4. ALWAYS end with '*** ANALYSIS COMPLETE ***' on its own line to signal you are done."
    )
    
    # -----------------------------------------------------------------
    # FIX: Pass the message objects directly. Do not strip them to text.
    # -----------------------------------------------------------------
    messages = [("system", system_prompt)] + state.get("messages",[])
    
    if state.get("whiteboard"):
        messages.append(HumanMessage(content=f"CURRENT WHITEBOARD:\n{state['whiteboard']}"))

    whiteboard_update = ""
    
    try:
        logger.debug("Invoking Analyst LLM with tools...")
        response = llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            logger.info(f"LLM decided to call tool: '{tool_name}'")
            
            tool_executed = False
            for tool in ANALYST_TOOLS:
                if getattr(tool, "name", None) == tool_name or getattr(tool, "__name__", None) == tool_name:
                    try:
                        logger.info(f"Executing tool '{tool_name}'...")
                        tool_result = tool.invoke(tool_args)
                        whiteboard_update = f"Analyst Insights (Tool: {tool_name}):\n{tool_result}\n\n*** ANALYSIS COMPLETE ***"
                        tool_executed = True
                        break
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        whiteboard_update = f"Analyst: Tool '{tool_name}' failed. Error: {str(e)[:200]}"
            
            if not tool_executed:
                logger.warning(f"Tool '{tool_name}' not found in registry.")
                whiteboard_update = f"Analyst: Tool '{tool_name}' not available.\n\n*** ANALYSIS COMPLETE ***"
        else:
            logger.info("LLM responded directly without tools.")
            whiteboard_update = f"Analyst Analysis:\n{response.content}\n\n*** ANALYSIS COMPLETE ***"
    
    except Exception as e:
        logger.error(f"Analyst execution failed: {e}", exc_info=True)
        whiteboard_update = f"Analyst: Analysis failed. Error: {str(e)[:200]}\n\n*** ANALYSIS COMPLETE ***"
    
    logger.info("Exiting Analyst Node.")
    return {
        "messages":[AIMessage(content="Analysis complete.")],
        "whiteboard": whiteboard_update,
        "next": AgentName.SUPERVISOR
    }

if __name__ == "__main__":
    logger.info("Testing Analyst Node...")
    mock_state = {
        "messages":[HumanMessage(content="What is the compound interest of $1000 at 5% over 10 years?")],
        "whiteboard": "Researcher: Found the formula is P(1+r)^t.",
        "next": AgentName.ANALYST
    }
    
    try:
        result = analyst_node(mock_state)
        logger.info(f"Updated Whiteboard:\n{result['whiteboard']}")
        logger.info(f"Next Node:\n{result['next']}")
    except Exception as e:
        logger.error(f"Error: {e}")