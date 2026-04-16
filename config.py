import os
from dotenv import load_dotenv

# ==========================================
# ⚙️ SYSTEM CONFIGURATION
# This file centralizes all constants and 
# environment loading to keep the code clean.
# ==========================================

# 1. Load environment parameters from the .env file
# (This step looks for OPENROUTER_API_KEY, LANGFUSE_PUBLIC_KEY, etc.)
load_dotenv()

# 2. Heuristics Thresholds
# Transactions with an amount greater than this are considered "High Value".
# The hackathon grades heavily on Economic Accuracy, so we must be extremely 
# careful with high-value transactions. We route these to the expensive LLM.
HIGH_VALUE_THRESHOLD = 1000.0 

# Transactions with an amount lower than this are generally safer or less 
# critical for Economic Accuracy. We can bypass LLM calls or use cheap LLMs.
LOW_VALUE_THRESHOLD = 5.0

# 3. Model Identifiers
# The cheap, fast model used by the Triage Agent to keep costs under the $40 budget.
CHEAP_MODEL_ID = "gpt-4o-mini"

# The powerful, capable model used for Deep Investigation (high value or confusing cases).
EXPENSIVE_MODEL_ID = "gpt-4o"

# OpenRouter Base URL standard
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Team identity for Langfuse tracking
TEAM_NAME = os.getenv("TEAM_NAME", "JuniorMindsProSolutions")
