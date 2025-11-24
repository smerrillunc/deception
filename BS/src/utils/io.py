import json, os, datetime
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, default=_json_default, indent=2)

def append_jsonl(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'a') as f:
        f.write(json.dumps(obj, default=_json_default) + '\n')

def _json_default(o):
    try:
        return o.__dict__
    except Exception:
        return str(o)
