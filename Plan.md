Ready for review
Select text to add comments on the plan
Reply Mirror — Fraud Detection Agent System: Initial Implementation Plan
Context
The repo is essentially empty (previous files were deleted). We need to build the full multi-agent fraud detection system described in CLAUDE.md from scratch. The system processes financial transactions and outputs suspected fraudulent transaction_ids. Data for the initial run is in data/ (80 transactions, 3 users, 829 GPS pings, 162 SMS threads, 8 mail threads).

Architecture Summary
Orchestrator + Tools pattern: A single LangChain agent (orchestrator) that invokes 5 specialist tools. Specialists are single-shot LLM calls wrapped in @tool — NOT nested agent loops. This cuts token cost and latency significantly.

Orchestrator (ReAct agent loop)
    ├── check_rules()      — deterministic, no LLM, always called first
    ├── check_profile()    — user salary/job vs. amount/merchant (LLM call)
    ├── check_geo()        — GPS ping speed-of-travel check (mostly deterministic + LLM synthesis)
    ├── check_behaviour()  — historical pattern deviations (rolling stats + LLM synthesis)
    └── check_comms()      — phishing SMS/mail correlation (feature extraction + LLM)
File Layout to Create
src/
├── __init__.py
├── config.py           — env loading, get_model() factory (ChatOpenAI via OpenRouter)
├── data_loader.py      — DataStore singleton, load_data(), IBAN↔user cross-reference
├── features.py         — cached BehaviourProfile, PhishingSignals, haversine, LocationIndex
├── rules.py            — check_rules(): deterministic checks, returns RuleVerdict
├── tracing.py          — Langfuse client init, make_session_id(), CallbackHandler
├── agents/
│   ├── __init__.py
│   ├── profile_agent.py   — @tool check_profile()
│   ├── geo_agent.py       — @tool check_geo()
│   ├── behaviour_agent.py — @tool check_behaviour()
│   ├── comms_agent.py     — @tool check_comms()
│   └── orchestrator.py    — create_agent() with all tools, @observe wrapper
└── run.py              — CLI: --input data/ --output out.txt
Key Design Decisions
DataStore Singleton Pattern
data_loader.py exposes a module-level _store and init_store(data_dir) / get_store(). Required because @tool-decorated functions can't take DataStore parameters (LangChain inspects signatures for tool schemas).

IBAN → User Cross-Reference
sender_iban in transactions matches iban in users.json. Build users_by_iban dict at load time, then users_by_biotag by iterating transactions:

for _, row in transactions.drop_duplicates("sender_id").iterrows():
    user = users_by_iban.get(row["sender_iban"])
    if user:
        users_by_biotag[row["sender_id"]] = user
Three EMP* sender_ids are employers with no profiles — specialists return UNCERTAIN for them.

SMS/Mail Association
Link by first name scan: scan each SMS/mail text for user first names. Build sms_by_user[biotag] and mails_by_user[biotag] at load time.

Specialist Tool Contracts
Each tool receives a compact JSON string and returns a JSON string:

Input: only the fields relevant to that specialist (not full transaction row)
Output: {"verdict": "FRAUD|LEGIT|UNCERTAIN", "confidence": 0.0-1.0, "reasoning": "..."}
Orchestrator Routing Logic (encoded in system prompt)
Transaction Type	Specialists to Call
Any	RuleEngine (always first; short-circuit if is_fraud=true)
in-person	+ GeoAgent + BehaviourAgent
e-commerce	+ ProfileAgent + CommsAgent + BehaviourAgent
transfer to new recipient	+ ProfileAgent + BehaviourAgent
direct_debit at unusual hour	+ BehaviourAgent + CommsAgent
Any off-hours (0–5 AM)	+ BehaviourAgent
RuleEngine Checks (in rules.py, no LLM)
amount > balance_before (balance_before = balance_after + amount)
sender_id == recipient_id
IBAN country prefix mismatch vs. user's registered country
Off-hours flag for known citizen senders (0–4 AM activity)
First-time recipient with amount > 3× user's typical amount
Verdict Aggregation (in orchestrator prompt)
RuleEngine is_fraud=true (confidence=1.0) → FRAUD immediately
2+ specialists with FRAUD + confidence > 0.7 → FRAUD
RuleEngine warning + 1 specialist FRAUD → FRAUD
Asymmetric cost: lean toward FRAUD when confidence > 0.5 and uncertain
Langfuse Tracing (v4.x)
get_client() auto-inits from env vars
@observe(name="orchestrate_transaction") wraps each transaction call
get_client().update_current_span(metadata={"session_id": sid}) tags every span
CallbackHandler() (no args) passed as LangChain callback to instrument LLM calls
Root start_as_current_observation in run.py groups the whole evaluation run
get_client().flush() at end of run
Build Order
src/config.py — model factory (blocks everything)
src/data_loader.py — DataStore + all cross-references
src/features.py — cached feature extraction + haversine
src/rules.py — deterministic rule engine + tests/test_rules.py
src/tracing.py — Langfuse setup
src/agents/profile_agent.py
src/agents/geo_agent.py
src/agents/behaviour_agent.py
src/agents/comms_agent.py
src/agents/orchestrator.py — wire all tools + @observe
src/run.py — CLI entrypoint
Output Safety Guardrails (in run.py)
Warn if fraud ratio < 5% (suspiciously low — might miss threshold)
Warn if fraud ratio > 50% (suspiciously high — likely over-flagging)
Print session_id to stdout for Langfuse cross-reference
Verification
# Install deps
source venv/bin/activate && pip install -r requirements.txt

# Run rule engine test
python -m pytest tests/test_rules.py -v

# Run full pipeline on sample data
python -m src.run --input data/ --output submissions/level_1.txt

# Check output
cat submissions/level_1.txt
Expected: output file with ≥1 and ≤40 transaction IDs (not all 80, not zero).