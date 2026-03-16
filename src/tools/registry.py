# src/tools/registry.py
# Centralized registry for tools used by different agents in the multi-agent system.
import os,sys
from pathlib import Path

# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.tools.search_tools import web_search
from src.tools.data_tools import calculate_math, profile_dataframe


# List of tools available to the Researcher
RESEARCHER_TOOLS = [web_search]

# List of tools available to the Analyst 
ANALYST_TOOLS = [calculate_math, profile_dataframe]