# Base tools and utilities for the multi-agent system.
#src/tools/base.py
import functools
from typing import Any, Callable

import os,sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Initializing base tools module...")

def safe_tool_execution(func: Callable) -> Callable:
    """
    A decorator for tools to catch exceptions and return them as string messages.
    This prevents graph crashes and allows the LLM to read the error and self-correct.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"ToolExecutionError in '{func.__name__}': {str(e)}"
            logger.warning(f"{error_msg}")
            return error_msg
    return wrapper

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---
    @safe_tool_execution
    def faulty_tool(x: int):
        return x / 0  # ZeroDivisionError

    @safe_tool_execution
    def working_tool(x: int):
        return x * 2

    logger.info(f"Working tool output: {working_tool(5)}")
    logger.info(f"Faulty tool output: {faulty_tool(5)}")