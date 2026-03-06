# Data-related tools for the multi-agent system.
# src/tools/data_tools.py
import pandas as pd
import numexpr as ne
import io, json
from langchain_core.tools import tool

import os,sys
from pathlib import Path

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.tools.base import safe_tool_execution
from src.utils.logger import get_logger
logger = get_logger()
logger.info("Initializing data tools module...")

@tool
@safe_tool_execution
def calculate_math(expression: str) -> str:
    """
    Evaluates a mathematical expression safely using numexpr.
    Useful for: financial calculations, statistics, percentages.
    Example expression: '((1024 * 3.14) / 2) + 100'
    """
    # ne.evaluate is significantly safer than python's built-in eval()
    result = ne.evaluate(expression)
    return str(result.item())

@tool
@safe_tool_execution
def profile_dataframe(json_data: str) -> str:
    """
    Takes a JSON string representation of data and returns a statistical summary.
    """
    df = pd.read_json(io.StringIO(json_data))
    summary = df.describe().to_string()
    return f"Data Profile Summary:\n{summary}"

if __name__ == "__main__":
    # --- DEBUG/TEST BLOCK ---    
    # Test Math
    math_expr = "100 * (1.05 ** 5)"
    logger.info(f"Math Test ({math_expr}): {calculate_math.invoke({'expression': math_expr})}")

    # Test Error Boundary inside Tool
    logger.info(f"Math Test (Invalid): {calculate_math.invoke({'expression': '100 / 0'})}")

    # Test DataFrame
    mock_data = '[{"sales": 100, "cost": 50}, {"sales": 150, "cost": 60}]'
    logger.info(f"DataFrame Test:\n{profile_dataframe.invoke({'json_data': mock_data})}")