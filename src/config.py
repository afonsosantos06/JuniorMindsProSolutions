"""Configuration: environment loading and model factory."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "gemini-3.1-flash-lite-preview"

TEAM_NAME = os.environ.get("TEAM_NAME", "the-eye")

# Thresholds
HIGH_VALUE_THRESHOLD = 1000.0
LOW_VALUE_THRESHOLD = 5.0


def get_model(temperature: float = 0.1) -> ChatOpenAI:
    """Return a ChatOpenAI instance pointing at OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )
