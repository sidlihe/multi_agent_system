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
    """
    Researcher node that gathers information using tools.
    Uses manual tool calling parsing (more stable with Groq + Llama-3.3)
    """
    logger.info("Entering Researcher Node.")

    llm = get_llm(temperature=0.15)  # low temperature → more consistent format

    # Build tool descriptions
    tool_descriptions = "\n".join(
        f"- {tool.name}: {getattr(tool, 'description', 'Search tool')}"
        for tool in RESEARCHER_TOOLS
    )

    system_prompt = f"""You are an expert Research Assistant.
    Your only goal is to collect accurate, up-to-date information to help answer questions.

    Available tools:
    {tool_descriptions}

    Rules:
    1. If you already know the answer or the whiteboard contains enough information → give a direct, concise final answer.
    2. If you need more information → call exactly ONE tool.
    3. To call a tool, output **ONLY** this exact format — nothing else:

    <function=tool_name>{{"query": "your precise search query"}}</function>

    Examples of correct usage:
    <function=web_search>{{"query": "current global stock market indices today"}}</function>
    <function=web_search>{{"query": "S&P 500 value right now"}}</function>

    VERY IMPORTANT:
    - Do NOT write JSON objects outside the tag
    - Do NOT add explanations before or after
    - Do NOT use <tool>, <call>, or any other tag
    - Do NOT output markdown, bullet points, or reasoning when calling a tool
    - Only when you are ready to give the final answer should you write normal text
    """

    messages = [("system", system_prompt),] + state.get("messages", [])

    if state.get("whiteboard"):
        messages.append(
            HumanMessage(content=f"Current whiteboard / known facts:\n{state['whiteboard']}")
        )

    logger.debug("Calling LLM (manual tool format mode)...")

    try:
        response = llm.invoke(messages)
        content = (response.content or "").strip()
        logger.debug(f"LLM raw output:\n{content[:60]}{'...' if len(content) > 60 else ''}")

    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        whiteboard_update = f"Researcher - LLM call failed: {str(e)[:180]}"
        return _return_state(state, whiteboard_update, next_agent=AgentName.SUPERVISOR)

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

    # Pattern 2: <function=xxx [ … ] (url)></function>  ← fallback for your original error
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

        tool_found = False
        for tool in RESEARCHER_TOOLS:
            if getattr(tool, "name", None) == tool_name:
                try:
                    result = tool.invoke({"query": query})
                    result_str = str(result).strip()
                    if len(result_str) > 1200:
                        result_str = result_str[:1150] + " … [truncated]"
                    whiteboard_update = (
                        f"Researcher searched: {query}\n"
                        f"Result:\n{result_str}"
                    )
                except Exception as exc:
                    whiteboard_update = f"Researcher tool failed: {str(exc)[:200]}"
                tool_found = True
                break

        if not tool_found:
            whiteboard_update = f"Researcher: unknown tool '{tool_name}'"

        next_agent = AgentName.RESEARCHER   # stay in researcher if we called tool

    else:
        # Normal answer — no tool call detected
        logger.info("LLM gave final answer (no tool call)")
        whiteboard_update = f"Researcher final answer:\n{content}"
        next_agent = AgentName.ANALYST      # or SUPERVISOR — adjust to your graph

    return _return_state(state, whiteboard_update, next_agent)


def _return_state(state, whiteboard_update: str, next_agent: str):
    """Helper to format consistent state return"""
    current_whiteboard = state.get("whiteboard", "")
    updated_whiteboard = current_whiteboard + ("\n\n" if current_whiteboard else "") + whiteboard_update

    return {
        "messages": state.get("messages", []) + [AIMessage(content=whiteboard_update)],
        "whiteboard": updated_whiteboard,
        "next": next_agent,
        # keep recursion_depth, sender, etc. if your graph uses them
        **{k: v for k, v in state.items() if k not in ["messages", "whiteboard", "next"]},
    }


if __name__ == "__main__":
    logger.info("Starting direct execution test for Researcher Node...")
    mock_state = {
        "messages": [HumanMessage(content="What is the current state of the global stock market today?")],
        "whiteboard": "",
        "next": AgentName.RESEARCHER,
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