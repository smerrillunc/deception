# poker_sim/snapshot.py
import json
import os
from typing import Dict, Any

def save_snapshot(snapshots_dir: str, stage: str, snap: Dict[str, Any]) -> str:
    os.makedirs(snapshots_dir, exist_ok=True)
    path = os.path.join(snapshots_dir, f"{stage}.json")
    with open(path, "w") as f:
        json.dump(snap, f, indent=2)
    return path

def load_snapshot(snapshot_path: str) -> Dict[str, Any]:
    with open(snapshot_path, "r") as f:
        return json.load(f)
