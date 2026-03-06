# src/agents/researcher.py
import os
import sys
import re
import json
from pathlib import Path
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.groq_client import get_llm
from src.tools.registry import RESEARCHER_TOOLS
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Initializing Researcher agent module...")

class AgentName(str, Enum):
    SUPERVISOR = settings.SUPERVISOR
    RESEARCHER = settings.RESEARCHER
    ANALYST = settings.ANALYST
    EVALUATOR = settings.EVALUATOR

def researcher_node(state):
    logger.info("Entering Researcher Node.")
    llm = get_llm()
    
    sys_msg = (
        "You are an expert Research Assistant.\n"
        "Your goal is to provide accurate, factual information to answer the user's request.\n"
        "Synthesize your findings clearly."
    )
    
    llm_with_tools = llm.bind_tools(RESEARCHER_TOOLS)
    messages = [("system", sys_msg)] + state.get("messages",[])
    
    if state.get("whiteboard"):
        messages.append(HumanMessage(content=f"CURRENT WHITEBOARD:\n{state['whiteboard']}"))

    whiteboard_update = ""

    try:
        logger.debug("Invoking LLM with bound tools...")
        response = llm_with_tools.invoke(messages)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"LLM decided to call tool: '{tool_name}' with args: {tool_args}")
            
            tool_executed = False
            for tool in RESEARCHER_TOOLS:
                if getattr(tool, "name", None) == tool_name or getattr(tool, "__name__", None) == tool_name:
                    logger.info(f"Executing '{tool_name}'...")
                    query_string = tool_args.get("query", str(tool_args))
                    
                    try:
                        tool_result = tool.invoke({"query": query_string})
                        logger.debug(f"Tool execution successful. Result length: {len(str(tool_result))}")
                        whiteboard_update = f"Researcher: Searched for '{query_string}'. Found data:\n{tool_result}"
                    except Exception as e:
                        logger.error(f"Tool execution raised exception: {e}")
                        whiteboard_update = f"Researcher: Tool execution failed: {str(e)[:200]}"
                    finally:
                        tool_executed = True
                    break
            
            if not tool_executed:
                logger.warning(f"LLM tried to call unknown tool: {tool_name}")
                whiteboard_update = f"Researcher: Attempted to use an unknown tool '{tool_name}'."
                
        else:
            logger.info("LLM responded directly without calling a tool.")
            whiteboard_update = f"Researcher Output: {response.content}"

    except Exception as e:
        error_text = str(e)
        pattern = r"<function=(?P<name>\w+)\s*(?P<args>\{.*?\})\s*</function>"
        m = re.search(pattern, error_text)

        # -----------------------------------------------------------------
        # FIX: Only log as a WARNING if it's the known Groq XML glitch. 
        # Do NOT print the stack trace unless it's a real, critical failure.
        # -----------------------------------------------------------------
        if m or "failed_generation" in error_text:
            logger.warning("Groq native tool parsing glitch detected. Engaging fallback local parser...")
            
            if m:
                tool_name = m.group("name")
                args_text = m.group("args")
                try:
                    tool_args = json.loads(args_text)
                except Exception:
                    tool_args = {"query": re.sub(r'^{|}$', '', args_text).strip(' "')}
            else:
                # Fallback if regex misses but we know it's a tool error
                tool_name = "web_search"
                tool_args = {"query": "Extracted from context"}

            logger.info(f"Executing fallback local tool '{tool_name}'...")
            
            tool_executed = False
            for tool in RESEARCHER_TOOLS:
                if getattr(tool, "name", None) == tool_name or getattr(tool, "__name__", None) == tool_name:
                    try:
                        tool_result = tool.invoke(tool_args)
                        whiteboard_update = f"Researcher: Searched via fallback. Found data:\n{tool_result}"
                        tool_executed = True
                        break
                    except Exception as e2:
                        logger.error(f"Fallback local tool execution failed: {e2}")
                        whiteboard_update = "Researcher: Attempted to search but both remote and local tool execution failed."
                        break
            
            if not tool_executed:
                whiteboard_update = f"Researcher: Fallback failed. Unknown tool '{tool_name}'."
        else:
            # THIS is where we print the hectic red stack trace, because it's an ACTUAL crash.
            logger.error(f"Critical LLM Execution failed: {e}", exc_info=True)
            whiteboard_update = "Researcher: Encountered a critical API error."

    logger.info("Exiting Researcher Node.")
    return {
        "messages": [HumanMessage(content="Research step completed. See whiteboard for details.")],
        "whiteboard": whiteboard_update,
        "next": AgentName.SUPERVISOR 
    }

if __name__ == "__main__":
    logger.info("Starting direct execution test for Researcher Node...")
    mock_state = {
        "messages":[HumanMessage(content="What is the current state of the global stock market today?")],
        "whiteboard": "",
        "next": AgentName.RESEARCHER,
        "recursion_depth": 0
    }
    result = researcher_node(mock_state)
    logger.info("=== FINAL NODE OUTPUT ===")
    logger.info(f"Next Node: {result.get('next')}")
    wb = result.get('whiteboard', '')
    logger.info(f"Updated Whiteboard:\n{wb[:800]}\n...[TRUNCATED]" if len(wb) > 800 else f"Updated Whiteboard:\n{wb}")