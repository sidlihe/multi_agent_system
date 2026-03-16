# src/agents/researcher.py
import os
import sys
import re
import json
from pathlib import Path
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.groq_client import get_llm
from src.tools.registry import RESEARCHER_TOOLS
from src.config.settings import settings,AgentName
from src.utils.logger import get_logger
from src.graph.completion import CompletionStatus, create_initial_completion_state, update_completion_state

logger = get_logger(__name__)
logger.info("Initializing Researcher agent module...")

def researcher_node(state):
    """
    Researcher node: Gathers information and tracks tool usage.
    """
    logger.info("Entering Researcher Node.")
    
    # Get or create completion state
    completion_state: CompletionStatus = state.get("completion_state")
    if not completion_state:
        completion_state = create_initial_completion_state()

    llm = get_llm(temperature=0.15)

    tool_descriptions = "\n".join(
        f"- {tool.name}: {getattr(tool, 'description', 'Search tool')}"
        for tool in RESEARCHER_TOOLS
    )

    system_prompt = f"""You are an expert Research Assistant.
    Your only goal is to collect accurate, up-to-date information to help answer questions.

    Available tools:
    {tool_descriptions}

    Rules:
    1. Check the whiteboard. If previous searches failed or returned irrelevant data, modify your search query. Do NOT repeat the exact same search.
    2. If you already know the answer or the whiteboard contains enough information → give a direct, concise final answer.
    3. To call a tool, output **ONLY** this exact format — nothing else:

    <function=tool_name>{{"query": "your precise search query"}}</function>

    Examples of correct usage:
    <function=web_search>{{"query": "current global stock market indices today"}}</function>
    <function=web_search>{{"query": "S&P 500 value right now"}}</function>

    VERY IMPORTANT:
    - Do NOT write JSON objects outside the tag
    - Do NOT add explanations before or after
    - Do NOT output markdown, bullet points, or reasoning when calling a tool
    """

    messages = [("system", system_prompt)] + state.get("messages", [])

    if state.get("whiteboard"):
        messages.append(
            HumanMessage(content=f"Current whiteboard / known facts:\n{state['whiteboard']}")
        )

    logger.debug("Calling LLM...")

    try:
        response = llm.invoke(messages)
        content = (response.content or "").strip()

    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        whiteboard_update = f"Researcher - LLM call failed: {str(e)[:180]}"
        return _return_state(state, whiteboard_update, AgentName.SUPERVISOR, completion_state)

    # ───────────────────────────────────────────────
    #  Parse possible tool call formats
    # ───────────────────────────────────────────────
    tool_call_detected = False
    tool_name = None
    query = None

    # Pattern 1: <function=xxx>{"query": "…"}</function>
    m1 = re.search(r'<function\s*=\s*(\w+)>\s*(\{.*?\})\s*</function>', content, re.DOTALL)
    if m1:
        tool_name = m1.group(1)
        try:
            args = json.loads(m1.group(2))
            query = args.get("query") or str(args)
            tool_call_detected = True
        except:
            pass

    # Pattern 2: Fallback parsing
    if not tool_call_detected:
        m2 = re.search(r'<function\s*=\s*(\w+)\s*\[(.*?)\].*?</function>', content, re.DOTALL)
        if m2:
            tool_name = m2.group(1)
            inner = m2.group(2).strip()
            qm = re.search(r'"query"\s*:\s*"([^"]+)"', inner)
            query = qm.group(1) if qm else inner.strip('" []')
            tool_call_detected = True

    if tool_call_detected and tool_name and query:
        logger.info(f"Tool call detected - {tool_name}  query: {query}")
        
        # Track tool usage
        completion_state.record_tool_usage(tool_name)

        tool_found = False
        for tool in RESEARCHER_TOOLS:
            if getattr(tool, "name", None) == tool_name:
                try:
                    result = tool.invoke({"query": query})
                    result_str = str(result).strip()
                    if len(result_str) > 1200:
                        result_str = result_str[:1150] + " … [truncated]"
                    whiteboard_update = f"Researcher searched: {query}\nResult:\n{result_str}"
                except Exception as exc:
                    whiteboard_update = f"Researcher tool failed: {str(exc)[:200]}"
                tool_found = True
                break

        if not tool_found:
            whiteboard_update = f"Researcher: unknown tool '{tool_name}'"

        next_agent = AgentName.RESEARCHER

    else:
        logger.info("LLM gave final answer (no tool call)")
        whiteboard_update = f"Researcher final summary:\n{content}"
        next_agent = settings.AgentName.ANALYST      

    return _return_state(state, whiteboard_update, next_agent, completion_state)


def _return_state(state, whiteboard_update: str, next_agent: str, completion_state: CompletionStatus):
    """Helper to format consistent state return.
    PRODUCTION FIX: Only return the new delta. Do not concatenate the old whiteboard, 
    otherwise LangGraph's state reducer will cause exponential duplication."""
    
    return {
        "messages": [AIMessage(content="Researcher updated the whiteboard.")],
        "whiteboard": whiteboard_update,  # <-- ONLY return the new update!
        "next": next_agent,
        "completion_state": completion_state,
        "recursion_depth": state.get("recursion_depth", 0) + 1,
        **{k: v for k, v in state.items() if k not in ["messages", "whiteboard", "next", "completion_state", "recursion_depth"]},
    }


if __name__ == "__main__":
    logger.info("Starting direct execution test for Researcher Node...")
    mock_state = {
        "messages": [HumanMessage(content="What is the current state of the global stock market today?")],
        "whiteboard": "",
        "next": settings.AgentName.RESEARCHER,
        "recursion_depth": 1
    }
    result = researcher_node(mock_state)
    #for self testing, print the result in a readable format
    print("\n" + "="*50)
    print(f"raw result dict:{result}")
    print("FINAL NODE OUTPUT")
    print("="*50)
    print(f"Next: {result.get('next')}")
    wb = result.get("whiteboard", "")
    if len(wb) > 900:
        print("Whiteboard (truncated):\n" + wb[:850] + "\n… [truncated]")
    else:
        print("Whiteboard:\n" + wb)