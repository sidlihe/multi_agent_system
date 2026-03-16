# src/agents/analyst.py
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import os, sys
from pathlib import Path


# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.utils.groq_client import get_llm
from src.tools.data_tools import calculate_math, profile_dataframe
from src.config.settings import settings, AgentName
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Initializing Analyst agent module...")

ANALYST_TOOLS = [calculate_math, profile_dataframe]

def analyst_node(state: dict) -> dict:
    logger.info("Entering Analyst Node.")
    llm = get_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools(ANALYST_TOOLS)
    
    system_prompt = (
        "You are a Senior Data Analyst.\n"
        "Your task is to analyze the data gathered in the CURRENT WHITEBOARD to answer the user's request.\n"
        "RULES:\n"
        "1. ONLY use the facts present in the whiteboard. Do not hallucinate external data.\n"
        "2. If you need to perform mathematical calculations, USE the `calculate_math` tool.\n"
        "3. If the whiteboard contains an '[ANALYSIS REJECTED]' note, correct your previous mistakes.\n"
        "4. Once your analysis is comprehensive, fully answers the prompt, and requires no further tool calls, ALWAYS end your response with exactly:\n"
        "*** ANALYSIS COMPLETE ***\n"
    )
    
    messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])
    current_whiteboard = state.get("whiteboard", "")
    
    if current_whiteboard:
        messages.append(HumanMessage(content=f"CURRENT WHITEBOARD:\n{current_whiteboard}"))

    whiteboard_update = ""
    
    try:
        logger.debug("Analyst analyzing...")
        response = llm_with_tools.invoke(messages)
        
        # ───────────────────────────────────────────────────────
        # IF THE LLM CALLS A TOOL
        # ───────────────────────────────────────────────────────
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")
            
            logger.info(f"Analyst calling tool: {tool_name}")
            
            tool_executed = False
            tool_result_str = ""
            
            for tool in ANALYST_TOOLS:
                if getattr(tool, "name", None) == tool_name or getattr(tool, "__name__", None) == tool_name:
                    try:
                        tool_result = tool.invoke(tool_args)
                        tool_result_str = str(tool_result)
                        whiteboard_update = f"Analyst Tool Result: {tool_name}({tool_args})\n{tool_result_str}\n[Note: Analyst must synthesize this data in the next step]"
                        tool_executed = True
                        break
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        tool_result_str = f"Error: {str(e)[:200]}"
                        whiteboard_update = f"Analyst Tool Error: {tool_name} failed."
            
            if not tool_executed:
                tool_result_str = "Error: Tool not found."
                whiteboard_update = f"Analyst: Tool '{tool_name}' not available."
                
            # THE MAGIC FIX: We return the LLM's actual response, followed by a ToolMessage.
            # This proves to the LLM that the tool was successfully executed!
            tool_msg = ToolMessage(
                tool_call_id=tool_id,
                name=tool_name,
                content=tool_result_str
            )
            
            logger.info("Exiting Analyst Node (Tool Called).")
            return {
                "messages": [response, tool_msg], # <--- Keeping the memory intact!
                "whiteboard": whiteboard_update,
                "next": AgentName.SUPERVISOR
            }
            
        # ───────────────────────────────────────────────────────
        # IF THE LLM WRITES A NORMAL TEXT RESPONSE
        # ───────────────────────────────────────────────────────
        else:
            logger.info("Analyst completed analysis.")
            content = (response.content or "").strip()
            
            # Ensure the completion marker is present if the LLM forgot it
            if "*** ANALYSIS COMPLETE ***" not in content:
                content += "\n\n*** ANALYSIS COMPLETE ***"
                
            whiteboard_update = f"Analyst Analysis:\n{content}"
            
            logger.info("Exiting Analyst Node.")
            return {
                "messages": [response], # Store the text response in history
                "whiteboard": whiteboard_update,
                "next": AgentName.SUPERVISOR
            }
    
    except Exception as e:
        logger.error(f"Analyst execution failed: {e}", exc_info=True)
        whiteboard_update = f"Analyst: Analysis failed. Error: {str(e)[:200]}\n[ANALYSIS REJECTED]"
        
        return {
            "messages": [AIMessage(content="Analyst encountered an error.")],
            "whiteboard": whiteboard_update,
            "next": AgentName.SUPERVISOR
        }

if __name__ == "__main__":
    logger.info("Testing Analyst Node...")