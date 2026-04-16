"""Langfuse tracing setup: client init, session IDs, callback handlers."""
import os

import ulid
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler

from src.config import TEAM_NAME

_langfuse_client: Langfuse | None = None


def init_langfuse() -> Langfuse:
    """Initialize and return the Langfuse client from environment variables."""
    global _langfuse_client
    _langfuse_client = Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    return _langfuse_client


def get_langfuse_client() -> Langfuse:
    """Return the initialized Langfuse client, initializing if needed."""
    global _langfuse_client
    if _langfuse_client is None:
        return init_langfuse()
    return _langfuse_client


def make_session_id(dataset_name: str = "run") -> str:
    """Generate a unique session ID for one evaluation run."""
    return f"{TEAM_NAME}-{dataset_name}-{ulid.new().str}"


def get_callback_handler(session_id: str | None = None) -> CallbackHandler:
    """
    Create a Langfuse CallbackHandler for instrumenting LangChain LLM calls.
    Pass session_id via metadata tagging after creation if needed.
    """
    if session_id:
        return CallbackHandler(session_id=session_id)
    return CallbackHandler()
