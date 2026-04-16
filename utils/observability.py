import os
import ulid
from langfuse import Langfuse

# ==========================================
# 📊 OBSERVABILITY & TRACKING (LANGFUSE)
# This module ensures that all our LLM calls 
# are correctly tracked, grouped by session, 
# and associated with our team name to 
# satisfy the Hackathon's tracking rules.
# ==========================================

def get_langfuse_client() -> Langfuse:
    """
    Initializes and returns the Langfuse client using the environment variables.
    The Reply Hackathon requires these specific variables to be set.
    """
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse")
    )

def generate_session_id(dataset_name: str = "dataset") -> str:
    """
    Generates a unique Session ID formatted specifically for the competition.
    Format: {TEAM_NAME}-{DATASET}-{ULID}
    
    Why is this important?
    1. It groups all transactions of a single dataset run into one logical "session" in the dashboard.
    2. The judges use this to track your $40 budget accurately.
    """
    team_name = os.getenv("TEAM_NAME", "JuniorMindsProSolutions")
    unique_id = ulid.new().str
    return f"{team_name}-{dataset_name}-{unique_id}"

# We instantiate a global client to be imported and used anywhere in the app
langfuse_client = get_langfuse_client()
