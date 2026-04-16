import os
import json
import uuid
import pandas as pd
from typing import Dict, Optional, Tuple

from agents import analyze_transaction
from config import DATASET_FOLDER, DATASETS_DIR, MAX_TRANSACTIONS, TEAM_NAME

def load_json_file(filepath: str) -> Optional[Dict]:
    """Load a JSON file safely. Return None if the file does not exist or is invalid."""
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Error loading {filepath}: {e}")
        return None

def load_dataset(dataset_name: str, datasets_dir: str = "datasets") -> Tuple[pd.DataFrame, Dict]:
    """Load transactions.csv plus optional metadata files for one dataset."""
    dataset_path = os.path.join(datasets_dir, dataset_name)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_name} not found.")
        return pd.DataFrame(), {}
    
    transactions_file = os.path.join(dataset_path, "transactions.csv")
    if not os.path.exists(transactions_file):
        print(f"❌ No transactions.csv found in {dataset_path}")
        return pd.DataFrame(), {}
    
    transactions_df = pd.read_csv(transactions_file)
    print(f"📊 Loaded {len(transactions_df)} transactions from {dataset_name}")
    
    metadata = {
        "users": load_json_file(os.path.join(dataset_path, "users.json")),
        "locations": load_json_file(os.path.join(dataset_path, "locations.json")),
        "mails": load_json_file(os.path.join(dataset_path, "mails.json")),
        "sms": load_json_file(os.path.join(dataset_path, "sms.json")),
    }
    
    loaded_files = [k for k, v in metadata.items() if v is not None]
    if loaded_files:
        print(f"📄 Loaded metadata: {', '.join(loaded_files)}")
    
    return transactions_df, metadata

def process_dataset(dataset_name: str) -> None:
    print(f"🚀 Team: {TEAM_NAME}")
    print(f"🚀 Starting processing for {dataset_name}...")

    session_id = f"{TEAM_NAME}-{dataset_name}-{uuid.uuid4().hex[:12]}"
    print(f"🆔 Langfuse session id: {session_id}")

    df, metadata = load_dataset(dataset_name, DATASETS_DIR)
    if df.empty:
        print(f"❌ Failed to load dataset {dataset_name}")
        return

    if MAX_TRANSACTIONS > 0:
        df = df.head(MAX_TRANSACTIONS)
        print(f"🧪 MAX_TRANSACTIONS enabled. Processing first {len(df)} rows.")

    flagged_transactions = []
    results = []

    for index, row in df.iterrows():
        transaction = row.to_dict()
        tx_id = transaction.get("transaction_id", f"row_{index}")

        print(f"Analyzing {tx_id} (Amount: {transaction.get('amount')})")
        analysis = analyze_transaction(transaction, metadata, session_id=session_id)

        print(f"   -> Decision: {analysis['decision']} | Confidence: {analysis['confidence']:.2f}")

        results.append({
            "transaction_id": tx_id,
            "decision": analysis["decision"],
            "confidence": analysis["confidence"],
            "reasoning": analysis["reasoning"],
        })

        if analysis["decision"] == "FRAUD":
            flagged_transactions.append(tx_id)

    out_file = f"submission_{dataset_name}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n✅ Finished processing {len(df)} transactions. Saved to {out_file}.")

    flagged_file = f"flagged_{dataset_name}.txt"
    with open(flagged_file, 'w') as f:
        for tx_id in flagged_transactions:
            f.write(f"{tx_id}\n")
    print(f"🚩 Flagged {len(flagged_transactions)} transactions. Saved to {flagged_file}.")


if __name__ == "__main__":
    process_dataset(DATASET_FOLDER)
