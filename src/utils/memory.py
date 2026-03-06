# Memory management utilities for the multi-agent system.
# src/utils/memory.py
from typing import List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage

import os,sys
from pathlib import Path
# Add the src directory to the system path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.utils.logger import get_logger
logger = get_logger()
logger.info("Initializing memory management module...")

# Rough estimate: 1 token ≈ 4 characters
def estimate_tokens(text: str) -> int:
    return len(text) // 4

def compress_history(messages: List[BaseMessage], max_tokens: int = 6000) -> List[BaseMessage]:
    """
    Checks if the message history exceeds max_tokens. 
    If it does, it drops the oldest messages (keeping the system prompt 
    and the most recent interactions).
    In a fully advanced setup, this would use an LLM to generate a summary.
    """
    total_tokens = sum(estimate_tokens(msg.content) for msg in messages if hasattr(msg, 'content'))
    
    if total_tokens <= max_tokens:
        return messages

    logger.info(f"[MEMORY] Context limit reached ({total_tokens}/{max_tokens}). Compressing...")
    
    # Keep the first message (usually system prompt/initial request)
    compressed = [messages[0]]
    
    # Find how many recent messages we can safely keep
    current_tokens = estimate_tokens(messages[0].content)
    keep_from_end = []
    
    for msg in reversed(messages[1:]):
        msg_tokens = estimate_tokens(msg.content)
        if current_tokens + msg_tokens > max_tokens:
            break
        keep_from_end.insert(0, msg)
        current_tokens += msg_tokens
        
    compressed.extend(keep_from_end)
    
    # Insert a system warning that history was truncated
    truncation_warning = AIMessage(content="[System: Older conversation history was truncated to save memory.]")
    compressed.insert(1, truncation_warning)
    
    return compressed

if __name__ == "__main__":
    # Create massive dummy messages
    large_text = "word " * 2000 # ~2000 tokens
    history =[
        HumanMessage(content="Initial Request: Build a system."),
        AIMessage(content=large_text),
        HumanMessage(content=large_text),
        AIMessage(content=large_text),
        HumanMessage(content="What was my last question?")
    ]
    
    logger.info(f"Original message count: {len(history)}")
    
    # Compress with a low token limit to force truncation
    compressed_history = compress_history(history, max_tokens=3000)
    
    logger.info(f"Compressed message count: {len(compressed_history)}")
    for i, msg in enumerate(compressed_history):
        # Print snippet of content
        logger.info(f"[{i}] {type(msg).__name__}: {msg.content[:50]}...")