multi_agent_system/
├── .env                       # API Keys (Groq, LangSmith, etc.)
├── pyproject.toml             # Dependencies (Poetry/pip)
├── README.md
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (LLM models, token limits)
│   ├── main.py                # Entry point (FastAPI or CLI)
│   │
│   ├── agents/                # Agent Definitions
│   │   ├── __init__.py
│   │   ├── supervisor.py      # The Router Agent
│   │   ├── researcher.py      # Research Agent
│   │   ├── analyst.py         # Data Analysis Agent
│   │   ├── evaluator.py       # Quality Control Agent
│   │   └── utils.py           # Helper functions for agents (e.g., summarizers)
│   │
│   ├── graph/                 # LangGraph Orchestration
│   │   ├── __init__.py
│   │   ├── state.py           # AgentState definition (Schema)
│   │   ├── workflow.py        # Graph construction (Nodes & Edges)
│   │   └── checkpoints.py     # Rollback & Persistence logic
│   │
│   ├── tools/                 # Tooling Layer
│   │   ├── __init__.py
│   │   ├── registry.py        # Tool registration & metadata mapping
│   │   ├── search_tools.py    # e.g., Tavily/Google Search
│   │   ├── data_tools.py      # e.g., Python REPL / Pandas
│   │   └── base.py            # Base classes for tool error handling
│   │
│   └── utils/
│       ├── __init__.py
│       ├── groq_client.py     # Centralized Groq client wrapper
│       ├── tracing.py         # LangSmith setup
│       └── memory.py          # Context window calculations
│
└── tests/
    ├── unit/                  # Test individual tools/agents
    └── integration/           # Test full graph workflows


---------------------------------------
>> # Create top-level files
>> ni .env -ItemType File
>> ni pyproject.toml -ItemType File
>> ni README.md -ItemType File
>> 
>> # Create src folder and subfolders
>> mkdir src
>> mkdir src\agents
>> mkdir src\graph
>> mkdir src\tools
>> mkdir src\utils
>> 
>> # Create tests folder and subfolders
>> mkdir tests
>> mkdir tests\unit
>> mkdir tests\integration
>>
>> # Create __init__.py files
>> ni src\__init__.py -ItemType File
>> ni src\agents\__init__.py -ItemType File
>> ni src\graph\__init__.py -ItemType File
>> ni src\tools\__init__.py -ItemType File
>> ni src\utils\__init__.py -ItemType File
>>
>> # Create main project files
>> ni src\config.py -ItemType File
>> ni src\main.py -ItemType File
>> 
>> # Agent files
>> ni src\agents\supervisor.py -ItemType File
>> ni src\agents\researcher.py -ItemType File
>> ni src\agents\analyst.py -ItemType File
>> ni src\agents\evaluator.py -ItemType File
>> ni src\agents\utils.py -ItemType File
>>
>> # Graph orchestration files
>> ni src\graph\state.py -ItemType File
>> ni src\graph\workflow.py -ItemType File
>> ni src\graph\checkpoints.py -ItemType File
>>
>> # Tooling files
>> ni src\tools\registry.py -ItemType File
>> ni src\tools\search_tools.py -ItemType File
>> ni src\tools\data_tools.py -ItemType File
>> ni src\tools\base.py -ItemType File
>>
>> # Utility files
>> ni src\utils\groq_client.py -ItemType File
>> ni src\utils\tracing.py -ItemType File
>> ni src\utils\memory.py -ItemType File