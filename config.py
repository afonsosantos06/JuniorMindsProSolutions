import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = os.getenv("MODEL_ID", "gemini-3.1-flash-lite-preview")

LANGFUSE_ENABLED = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

TEAM_NAME = os.getenv("TEAM_NAME", "JuniorMindsProSolutions")
DATASET_FOLDER = "The Truman Show - train"
DATASETS_DIR = os.getenv("DATASETS_DIR", "datasets")
MAX_TRANSACTIONS = int(os.getenv("MAX_TRANSACTIONS", "0"))
