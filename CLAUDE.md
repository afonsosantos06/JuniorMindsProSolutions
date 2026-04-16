# Reply Mirror — Fraud Detection Agent System

> **Challenge:** Reply Code Challenge 2026 — "Reply Mirror"
> **Goal:** Detect fraudulent financial transactions in the digital metropolis of Reply Mirror using a cooperative multi-agent system.
> **Stack:** Python 3.10–3.13, LangChain, OpenRouter (gpt-4o-mini), Langfuse (observability).

---

## 1. Problem Recap

In 2087, the financial institution **MirrorPay** processes billions of transactions. **Mirror Hackers** inject fraudulent transactions that blend with legitimate activity. Their patterns evolve across five challenge levels (new merchants, shifted hours, new geographies, varying amounts, deceptive sequences).

Our job, as **The Eye**, is to build an agent-based system that, for every transaction in an evaluation set, decides: **legitimate** or **fraudulent**, and outputs the fraudulent `transaction_id`s to a plain ASCII file (one per line).

**Asymmetric cost:**
- False positive (block legit tx) → economic + reputational cost.
- False negative (miss fraud) → direct financial damage.
- Output is invalid if: none reported, all reported, or <15% of frauds caught.

**Scoring:**
- **Accuracy** (primary): balanced fraud detection vs. low false-positive rate.
- **Additional metrics:** cost, latency, token efficiency (tracked via Langfuse session IDs).

---

## 2. Input Data

All inputs live in the project root (or a `data/` subfolder per level).

| File | Content | Notes |
|---|---|---|
| `transactions.csv` | Financial transactions | Columns: `transaction_id, sender_id, recipient_id, transaction_type, amount, location, payment_method, sender_iban, recipient_iban, balance_after, description, timestamp`. `transaction_type ∈ {transfer, in-person, e-commerce, direct_debit, withdrawal}`. |
| `users.json` | Citizen profiles | Name, birth year, salary, job, IBAN, residence (city + lat/lng), behavioural `description` (in native language) hinting at phishing susceptibility, travel habits, etc. |
| `locations.json` | GPS pings | `biotag` (citizen ID), `timestamp`, `lat`, `lng`, `city`. Used for "was this person actually there?" checks. |
| `sms.json` | SMS threads | Full textual thread per entry. Phishing, city alerts, delivery notices, etc. |
| `mails.json` | Email threads | Full raw email (headers + HTML body). Bills, invoices, potential phishing. |

**Cross-reference keys:**
- `sender_id` / `recipient_id` ↔ `biotag` (in locations) ↔ implicit user identity (via IBAN in `users.json`).
- IBAN is the hardest link: `users.json[i].iban` matches `sender_iban` / `recipient_iban` in transactions.

---

## 3. Output Format

A plain ASCII file, one `transaction_id` per line, listing only **suspected fraudulent** transactions.

```
4a92ab00-8a27-4623-ab1d-56ac85fcd6b0
8830a720-ff34-4dce-a578-e5b8006b2976
```

---

## 4. System Architecture — Orchestrator Pattern

We use the **"Agents as Tools"** pattern from Tutorial 03: a single **Orchestrator Agent** coordinates several **Specialist Agents**, each exposed to the orchestrator as a `@tool`-decorated function. The orchestrator decides which specialists to consult for each transaction and synthesizes their verdicts.

```
                  ┌──────────────────────────────────┐
                  │       Orchestrator Agent         │
                  │  (decides which specialists to   │
                  │   call, aggregates verdicts,     │
                  │   emits final fraud/legit label) │
                  └──────────────┬───────────────────┘
                                 │
        ┌────────────┬───────────┼───────────┬────────────┐
        ▼            ▼           ▼           ▼            ▼
   ┌─────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐
   │Profile  │ │Geo/      │ │Behav.  │ │Comms     │ │Rule       │
   │Agent    │ │Location  │ │Pattern │ │Agent     │ │Engine     │
   │         │ │Agent     │ │Agent   │ │(SMS+Mail)│ │(determ.)  │
   └─────────┘ └──────────┘ └────────┘ └──────────┘ └───────────┘
        │            │           │           │            │
        ▼            ▼           ▼           ▼            ▼
    users.json  locations    tx history   sms.json    hard rules
                             per sender   mails.json  (IBAN check,
                                                       amount>balance,
                                                       etc.)
```

### 4.1 Specialist Agents

Each specialist is an independent `create_agent(...)` call with a focused system prompt, exposed via `@tool` so the orchestrator can invoke it.

| Specialist | Role | Key signals |
|---|---|---|
| **ProfileAgent** | Is the amount / merchant consistent with this user's salary, job, habits? | `users.json` description, salary, residence city. |
| **GeoAgent** | Was the user plausibly at the transaction location? Speed-of-travel check between consecutive pings. | `locations.json` + transaction `location` + `timestamp`. |
| **BehaviourAgent** | Does the tx break the sender's historical pattern (time-of-day, merchant category, typical amount)? | Rolling statistics over `transactions.csv` per `sender_id`. |
| **CommsAgent** | Did the sender recently receive phishing? Does the tx correlate with a suspicious SMS/email? | `sms.json`, `mails.json` filtered by recipient. |
| **RuleEngine** | Hard deterministic checks (fast, cheap, no LLM). | Amount > balance, IBAN format mismatch, sender == recipient, impossible timestamps, etc. |

> **Why a RuleEngine as a "tool"?** The challenge requires agent-based solutions and penalises fully-deterministic approaches, but rules are a legitimate *tool* the agent chooses to consult. They handle the easy 80% cheaply; the LLM specialists handle the ambiguous 20%.

### 4.2 Orchestrator Responsibilities

1. Receive a batch of transactions (or one at a time).
2. Run the cheap **RuleEngine** first — if it catches an obvious fraud, short-circuit and skip LLM calls.
3. For ambiguous transactions, consult 1–N specialists based on which signals are most informative (e.g., in-person payments → GeoAgent; e-commerce → ProfileAgent + CommsAgent).
4. Combine verdicts (weighted vote, or a small "judge" prompt) into a binary label.
5. Emit the `transaction_id` to the output file if fraudulent.

### 4.3 Why this architecture

- **Separation of concerns** — each agent has one job and a tight prompt.
- **Cost control** — the orchestrator avoids calling every specialist on every tx.
- **Adaptability** — challenge levels shift attack patterns; we can swap or reweight specialists per level without rewriting the core.
- **Observability** — every specialist call is a nested trace under the orchestrator's Langfuse trace.

---

## 5. Resource Management (Langfuse)

Per Tutorial 04, **all** cost/token tracking goes through Langfuse session IDs.

- Wrap the orchestrator entrypoint with `@observe()`.
- Create a `CallbackHandler()` inside the decorated function and attach it to every `model.invoke(...)` call (including inside specialists).
- Generate one `session_id` per evaluation run: `f"{TEAM_NAME}-{ulid.new().str}"`.
- Tag the trace: `langfuse_client.update_current_trace(session_id=session_id)`.
- `langfuse_client.flush()` at the end of the run.

A single `session_id` groups *every* LLM call across all agents for that evaluation, so the judges can query aggregate cost/tokens/latency in one shot.

**Cost discipline:**
- Cache per-user feature computations (geo speed, rolling tx stats) so specialists don't recompute.
- Short system prompts. Pass only the **relevant** slice of data to each specialist (not the whole JSON file).
- Use `temperature=0.1` or lower for decision agents; only the orchestrator's synthesis may need slightly higher.

---

## 6. Project Layout

```
reply-mirror/
├── Claude.md                    ← this file
├── .env                         ← OPENROUTER_API_KEY, LANGFUSE_*, TEAM_NAME
├── requirements.txt
├── data/
│   ├── level_1/
│   │   ├── transactions.csv
│   │   ├── users.json
│   │   ├── locations.json
│   │   ├── sms.json
│   │   └── mails.json
│   └── level_2/ ...
├── src/
│   ├── __init__.py
│   ├── config.py                ← env loading, model factory
│   ├── data_loader.py           ← load + index CSV/JSON, build lookup dicts
│   ├── features.py              ← cached per-user feature extraction
│   ├── rules.py                 ← deterministic RuleEngine checks
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── profile_agent.py
│   │   ├── geo_agent.py
│   │   ├── behaviour_agent.py
│   │   ├── comms_agent.py
│   │   └── orchestrator.py      ← composes all specialists as @tools
│   ├── tracing.py               ← Langfuse client, @observe wrappers, session IDs
│   └── run.py                   ← CLI: python -m src.run --input data/level_1 --output out.txt
├── notebooks/                   ← exploration, not part of the submission
│   └── eda.ipynb
└── tests/
    └── test_rules.py            ← unit tests for deterministic checks
```

---

## 7. Execution

```bash
# one-time
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# per evaluation level
python -m src.run \
    --input data/level_1 \
    --output submissions/level_1.txt
```

`run.py` prints the generated `session_id` so we can cross-reference it in Langfuse.

---

## 8. Development Workflow

1. **EDA first** — before any agent, understand the data in `notebooks/eda.ipynb`: class balance, fraud signatures in the example, language distribution in sms/mails.
2. **Build the RuleEngine** — catches obvious cases and gives a baseline fraud list. Measure: how many frauds in the example is it already catching?
3. **Build one specialist at a time** — start with GeoAgent (clearest signal), then ProfileAgent. Test each in isolation.
4. **Wire the orchestrator** — start with a *sequential* orchestrator (ask every specialist every time), measure cost & accuracy, then optimise to *selective* consultation.
5. **Langfuse from day one** — don't bolt tracing on at the end; add `@observe()` + `CallbackHandler` from the first agent.
6. **Level-by-level adaptation** — after each evaluation submission, review Langfuse traces of misclassified transactions and adjust prompts/weights before the next level.

---

## 9. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Judges penalise deterministic solutions | RuleEngine is a *tool*, not the whole system; LLM agents always have final say on ambiguous tx. |
| Token cost blows up | Pass slices, not full files, to specialists. Cache features. Short-circuit on RuleEngine hits. |
| >85% false-negative floor invalidates output | Keep a minimum recall safety net: if we flag <15%, lower the decision threshold before writing. |
| Over-flagging (all tx = fraud) invalidates output | Symmetric guardrail: cap flagged ratio at a sane upper bound (e.g. 40%). |
| Pattern shifts between levels | Specialists are prompt-driven, so we can re-prompt per level without code changes. |
| Langfuse Python 3.14 incompatibility | Pin Python 3.10–3.13 in the venv. |

---

## 10. Environment Variables (`.env`)

```
OPENROUTER_API_KEY=sk-or-...
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=the-eye
```

---

## 11. References

- Tutorial 01 — Agent creation (`create_agent`, system prompts).
- Tutorial 02 — Adding tools (`@tool` decorator, docstrings are the tool's interface).
- Tutorial 03 — Multi-agent orchestration (agents-as-tools pattern).
- Tutorial 04 — Langfuse tracing (`@observe`, `CallbackHandler`, session IDs).
- Problem statement — `AIAgentChallenge-ProblemStatement16April.pdf`.