# src/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

    GROQ_API_KEY: str
    GROQ_MODEL: str 

    TAVILY_API_KEY: str | None = None

    MAX_CONTEXT_TOKENS: int = 6000
    MAX_ITERATIONS: int = 10  # Allow up to 10 supervisor->agent cycles to prevent premature timeout
    TEMPERATURE: float = 0.3

    # Agent Names
    SUPERVISOR: str = "Supervisor"
    RESEARCHER: str = "Researcher"
    ANALYST: str = "Analyst"
    EVALUATOR: str = "Evaluator"

settings = Settings()