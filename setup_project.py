import os

# List of directories to create
directories = [
    "src",
    "src/agents",
    "src/graph",
    "src/tools",
    "src/utils",
    "tests",
    "tests/unit",
    "tests/integration",
]

# List of files to create
files = [
    ".env",
    "pyproject.toml",
    "README.md",

    "src/__init__.py",
    "src/agents/__init__.py",
    "src/graph/__init__.py",
    "src/tools/__init__.py",
    "src/utils/__init__.py",

    "src/config.py",
    "src/main.py",

    "src/agents/supervisor.py",
    "src/agents/researcher.py",
    "src/agents/analyst.py",
    "src/agents/evaluator.py",
    "src/agents/utils.py",

    "src/graph/state.py",
    "src/graph/workflow.py",
    "src/graph/checkpoints.py",

    "src/tools/registry.py",
    "src/tools/search_tools.py",
    "src/tools/data_tools.py",
    "src/tools/base.py",

    "src/utils/groq_client.py",
    "src/utils/tracing.py",
    "src/utils/memory.py",
]

# Create directories
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Skipped (exists): {directory}")

# Create files
for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
        print(f"Created file: {file}")
    else:
        print(f"Skipped (exists): {file}")

print("\nProject structure setup complete.")