# src/graph/completion.py
"""
Structured completion tracking to replace string-based markers.
This ensures consistent agent behavior and prevents infinite loops.
"""

import json
from typing import Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class CompletionStatus:
    """Tracking structure for workflow completion state"""
    stage: str  # "RESEARCH", "ANALYSIS", "EVALUATION", "DONE"
    confidence: float  # 0.0 to 1.0
    is_complete: bool
    last_evaluator_feedback: str = ""
    refinement_attempts: int = 0
    max_refinements_allowed: int = 2
    tools_used: Dict[str, int] = None  # Track which tools and how many times
    
    def __post_init__(self):
        if self.tools_used is None:
            self.tools_used = {}
    
    def to_json_marker(self) -> str:
        """Serialize to a JSON string for embedding in whiteboard"""
        return f"\n[COMPLETION_STATE:{json.dumps(self.__dict__)}]"
    
    @staticmethod
    def from_whiteboard(whiteboard: str) -> Optional['CompletionStatus']:
        """Extract completion state from whiteboard if present"""
        try:
            import re
            match = re.search(r'\[COMPLETION_STATE:(.*?)\]', whiteboard)
            if match:
                data = json.loads(match.group(1))
                return CompletionStatus(
                    stage=data.get('stage', 'RESEARCH'),
                    confidence=data.get('confidence', 0.0),
                    is_complete=data.get('is_complete', False),
                    last_evaluator_feedback=data.get('last_evaluator_feedback', ''),
                    refinement_attempts=data.get('refinement_attempts', 0),
                )
        except:
            pass
        return None
    
    def should_force_finish(self) -> bool:
        """Check if we should force workflow termination"""
        if self.is_complete and self.confidence >= 0.75:
            return True
        if self.refinement_attempts >= self.max_refinements_allowed:
            return True
        return False
    
    def should_attempt_refinement(self) -> bool:
        """Check if we should try to refine rather than abandon"""
        return self.refinement_attempts < self.max_refinements_allowed
    
    def record_tool_usage(self, tool_name: str):
        """Track which tools were used"""
        self.tools_used[tool_name] = self.tools_used.get(tool_name, 0) + 1
    
    def get_refinement_hint(self) -> str:
        """Generate a hint about what to refine based on feedback"""
        feedback_lower = self.last_evaluator_feedback.lower()
        if "application process" in feedback_lower:
            return "Focus on adding specific details about the application process."
        elif "calculation" in feedback_lower:
            return "Verify numerical calculations and recalculate if needed."
        elif "missing" in feedback_lower:
            return "Ensure all parts of the user's multi-part question are answered."
        elif "specific" in feedback_lower:
            return "Add more specific details and concrete examples."
        elif feedback_lower:
            return f"Address the feedback: {self.last_evaluator_feedback[:100]}"
        return "Review the feedback and improve the response."


def create_initial_completion_state() -> CompletionStatus:
    """Create initial completion state for a new workflow"""
    return CompletionStatus(
        stage="RESEARCH",
        confidence=0.0,
        is_complete=False,
        last_evaluator_feedback="",
        refinement_attempts=0,
        tools_used={}
    )


def update_completion_state(
    state: CompletionStatus,
    stage: Optional[str] = None,
    confidence: Optional[float] = None,
    is_complete: Optional[bool] = None,
    feedback: Optional[str] = None,
    tool_used: Optional[str] = None
) -> CompletionStatus:
    """Update completion state with new information"""
    if stage is not None:
        state.stage = stage
    if confidence is not None:
        state.confidence = confidence
    if is_complete is not None:
        state.is_complete = is_complete
    if feedback is not None:
        state.last_evaluator_feedback = feedback
        state.refinement_attempts += 1
    if tool_used is not None:
        state.record_tool_usage(tool_used)
    
    return state
