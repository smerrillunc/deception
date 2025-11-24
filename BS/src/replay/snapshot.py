import json, os, time
from ..utils.io import write_json, ensure_dir

def save_full_snapshot(path, game_state: dict, meta: dict):
    ensure_dir(os.path.dirname(path))
    bundle = {'meta': meta, 'state': game_state, 'timestamp': time.time()}
    write_json(bundle, path)
    return path
