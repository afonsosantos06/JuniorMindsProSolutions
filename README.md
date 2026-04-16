# JuniorMindsProSolutions

Minimal multi-agent fraud detection setup.

Core implementation files:
- `config.py`
- `agents.py`
- `main.py`

What it does:
- Loads one dataset (`transactions.csv` + optional metadata json files)
- Runs a simple multi-agent fraud analysis per transaction:
	- Triage agent
	- Context-check agent
	- Orchestrator agent
- Writes:
	- `submission_<dataset>.csv`
	- `flagged_<dataset>.txt`

Run:

```bash
pip install -r requirements.txt
python main.py
```

Environment variables:
- `OPENROUTER_API_KEY` (required)
- `MODEL_ID` (optional, default `gpt-4o-mini`)
- `DATASETS_DIR` (optional, default `datasets`)
- `MAX_TRANSACTIONS` (optional, default `0` meaning all rows)
