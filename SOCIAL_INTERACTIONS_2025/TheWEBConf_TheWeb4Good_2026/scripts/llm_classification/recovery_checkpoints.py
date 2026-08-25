# recover_checkpoints.py
import json
import pandas as pd
from pathlib import Path

def recover_checkpoint_data(output_dir):
    """Recover successful classifications from checkpoint files"""
    all_success = []
    all_failed = []
    
    checkpoint_files = list(output_dir.glob("checkpoint_chunk_*.json"))
    
    for checkpoint_file in checkpoint_files:
        print(f"Processing {checkpoint_file.name}...")
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
        
        all_success.extend(data.get("success", []))
        all_failed.extend(data.get("failed", []))
        
        print(f"  - {len(data.get('success', []))} success")
        print(f"  - {len(data.get('failed', []))} failed")
    
    return all_success, all_failed

if __name__ == "__main__":
    # Run recovery
    output_dir = Path("/home/ethel/Documents/RESEARCH/SOCIAL_INTERACTIONS_2025/TheWEBConf_TheWeb4Good_2026/working_data/classification_chunks/")
    success, failed = recover_checkpoint_data(output_dir)

    print(f"\n📊 Total recovered: {len(success)} success, {len(failed)} failed")

    # Save recovered data
    if success:
        success_df = pd.DataFrame(success)
        success_df.to_csv(output_dir / "recovered_success.csv", index=False)
        print(f"💾 Saved recovered successes to recovered_success.csv")

    if failed:
        failed_df = pd.DataFrame(failed)
        failed_df.to_csv(output_dir / "recovered_failed.csv", index=False)
        print(f"💾 Saved recovered failures to recovered_failed.csv")