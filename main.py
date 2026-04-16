import os
import json
import pandas as pd
from typing import List, Dict

from config import TEAM_NAME
from utils.observability import langfuse_client, generate_session_id
from workflow import app

# ==========================================
# 🚦 HACKATHON ENTRY POINT
# Put this together with your provided datasets.
# Example usage: python main.py
# ==========================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load your challenge CSV dataset."""
    if not os.path.exists(filepath):
        print(f"⚠️ Mocking dataset since {filepath} does not exist.")
        # Create a mock dataset if it doesn't exist to test the skeleton
        return pd.DataFrame([
            {"transaction_id": "T001", "amount": 2.50, "location": "Lisbon"}, # Should hit heuristic
            {"transaction_id": "T002", "amount": 150.0, "location": "Unknown"}, # Should hit Triage
            {"transaction_id": "T003", "amount": 5000.0, "location": "Offshore"} # Should hit Deep Investigator
        ])
    return pd.read_csv(filepath)

def process_dataset(filepath: str, dataset_name: str) -> None:
    print(f"🚀 Starting processing for {dataset_name}...")
    
    # 1. Generate the crucial Session ID for grading
    session_id = generate_session_id(dataset_name)
    print(f"🆔 Session ID for this run: {session_id}")
    
    # 2. Load the data
    df = load_dataset(filepath)
    results = []
    
    # 3. Process row by row
    # (For 6-hour hackathons, synchronous row-by-row is safer for avoiding rate-limits)
    for index, row in df.iterrows():
        transaction = row.to_dict()
        tx_id = transaction.get("transaction_id", f"row_{index}")
        
        print(f"Analyzing {tx_id} (Amount: {transaction.get('amount')})")
        
        # Initial State
        initial_state = {
            "transaction": transaction,
            "session_id": session_id,
            "heuristic_passed": False,
            "final_decision": None,
            "confidence_score": 0.0,
            "reasoning": "",
            "escalated": False
        }
        
        # Invoke the LangGraph workflow
        final_state = app.invoke(initial_state)
        
        print(f"   -> Decision: {final_state['final_decision']} | Confidence: {final_state['confidence_score']:.2f}")
        
        results.append({
            "transaction_id": tx_id,
            "decision": final_state['final_decision'],
            "confidence": final_state['confidence_score'],
            "reasoning": final_state['reasoning'],
        })
        
    
    # 4. Save results for the Reply submission portal
    out_file = f"submission_{dataset_name}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n✅ Finished processing {len(df)} transactions. Saved to {out_file}.")
    
    # 5. VERY IMPORTANT: Flush Langfuse traces to ensure they reach the Reply dashboard
    langfuse_client.flush()
    print("✅ All telemetry data flushed to Langfuse.")

if __name__ == "__main__":
    # Simulate processing the first Training Dataset
    process_dataset("data/dataset_1.csv", "dataset_1")
    
    # Once the real data drops, duplicate the call:
    # process_dataset("data/dataset_2.csv", "dataset_2")
