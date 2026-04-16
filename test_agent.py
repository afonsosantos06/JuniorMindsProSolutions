import sys
from pathlib import Path
repo_root = Path().resolve()
sys.path.insert(0, str(repo_root))

from src.agents.orchestrator import _build_agent
print("trying to build agent")
agent = _build_agent()
print("agent built:", agent)
