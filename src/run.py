"""CLI entrypoint: python -m src.run --input data/ --output submissions/level_1.txt"""
import argparse
import sys
import os
from pathlib import Path


# Allow running as a script: `python src/run.py ...`.
# When executed this way, Python sets sys.path[0] to the `src/` directory,
# which prevents `import src.*` from resolving unless we add the repo root.
if __package__ is None:  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(
        description="Reply Mirror fraud detection pipeline"
    )
    parser.add_argument("--input", required=True, help="Path to data directory")
    parser.add_argument("--output", required=True, help="Output file path for fraud IDs")
    parser.add_argument(
        "--no-tracing", action="store_true",
        help="Disable Langfuse tracing (useful for local testing)"
    )
    args = parser.parse_args()

    # --- Initialize data ---
    print(f"Loading data from: {args.input}")
    from src.data_loader import init_store, get_store
    init_store(args.input)
    store = get_store()

    total = len(store.transactions)
    print(f"Loaded {total} transactions, {len(store.users_by_biotag)} citizen users")

    # --- Initialize tracing ---
    session_id = None
    callback_handler = None

    if not args.no_tracing:
        try:
            from src.tracing import init_langfuse, make_session_id, get_callback_handler
            init_langfuse()
            data_name = Path(args.input).name or "run"
            session_id = make_session_id(data_name)
            callback_handler = get_callback_handler(session_id)
            print(f"Session ID: {session_id}")
        except Exception as e:
            print(f"[WARN] Langfuse tracing unavailable: {e}")

    # --- Process transactions ---
    from src.agents.orchestrator import orchestrate_transaction

    fraud_ids = []
    print(f"\nProcessing {total} transactions...")

    for i, row in store.transactions.iterrows():
        tx = row.to_dict()
        tx_id, is_fraud = orchestrate_transaction(tx, callback_handler=callback_handler)

        if is_fraud:
            fraud_ids.append(tx_id)
            print(f"  [{i+1:03d}/{total}] FRAUD   : {tx_id}")
        else:
            print(f"  [{i+1:03d}/{total}] legit   : {tx_id}")

    # --- Safety guardrails ---
    ratio = len(fraud_ids) / total if total > 0 else 0
    print(f"\nResults: {len(fraud_ids)}/{total} flagged as fraud ({ratio:.1%})")

    if ratio < 0.05:
        print("[WARN] Very low fraud rate — may fall below the 15% recall floor.")
    if ratio > 0.50:
        print("[WARN] Very high fraud rate — may be over-flagging.")

    # --- Write output ---
    output_path = Path(args.output)
    if not output_path.parent.exists():
        os.makedirs(output_path.parent, exist_ok=True)
    output_path.write_text("\n".join(fraud_ids) + ("\n" if fraud_ids else ""))
    print(f"Output written to: {args.output}")

    # --- Flush Langfuse ---
    if session_id:
        try:
            from src.tracing import get_langfuse_client
            get_langfuse_client().flush()
            print(f"Langfuse flushed. Session: {session_id}")
        except Exception as e:
            print(f"[WARN] Langfuse flush failed: {e}")


if __name__ == "__main__":
    main()
