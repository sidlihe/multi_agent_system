from pydantic import BaseModel, Field
from typing import List,Any,Optional

class LLMUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class LLMResponse(BaseModel):
    data: Any
    # latency: float
    usage: Optional[LLMUsage] = None
    model: Optional[str] = None

class ResearchResult(BaseModel):
    topic: str
    findings: List[str]
    sources: List[str]


class AnalysisResult(BaseModel):
    summary: str
    key_points: List[str]
    confidence: float


class EvaluationResult(BaseModel):
    verdict: str
    score: float
    reasoning: str

class SearchInput(BaseModel):
    query: str = Field(description="The exact search query to look up on the web.")