import random, numpy as np, torch
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

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def load_model_and_tokenizer(model_name, max_seq_length=4000, device_map="auto"):
    import os
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        max_seq_length=max_seq_length,
        device_map=device_map,
        load_in_4bit=True,
        fix_tokenizer=True,
        offload_folder="/playpen-ssd/smerrill/offload", 
    ) 
    if 'llama' in model_name.lower():
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = "llama-3.1",
        )
    return model, tokenizer
