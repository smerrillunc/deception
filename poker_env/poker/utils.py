# poker_sim/utils.py
import json
import re
import numpy as np
import torch
import os
from typing import Any

def safe_parse_json(raw_text: str):
    try:
        m = re.search(r"\{.*\}", raw_text, flags=re.S)
        if not m:
            raise ValueError("No JSON block found")
        js_text = m.group()
        try:
            return json.loads(js_text)
        except Exception:
            tmp = js_text.replace('\u201c', '"').replace('\u201d', '"')
            tmp = tmp.replace('\u2018', "'").replace('\u2019', "'")
            keys = ["reasoning", "text", "action", "amount"]
            extracted = {}
            for key in keys:
                pat = re.compile(r'"?%s"?\s*:\s*(?P<val>(?:"(?:\\.|[^"])*")|(?:\'(?:\\.|[^\'])*\')|[^,}\n]+)' % re.escape(key), flags=re.S)
                m2 = pat.search(tmp)
                if m2:
                    v = m2.group('val').strip()
                    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                        v = v[1:-1]
                        v = v.replace('\\"', '"').replace("\\'", "'")
                    extracted[key] = v
            if extracted:
                if 'amount' in extracted:
                    try:
                        extracted['amount'] = int(re.sub(r'[^0-9-]', '', extracted['amount']))
                    except Exception:
                        extracted['amount'] = 0
                return extracted
            tmp2 = re.sub(r"(?<=\\w)'(?=\\w)", "__APOS__", tmp)
            tmp2 = re.sub(r",\s*}\s*$", "}", tmp2)
            tmp2 = tmp2.replace("'", '"')
            tmp2 = tmp2.replace("__APOS__", "'")
            return json.loads(tmp2)
    except Exception as e:
        print("COULD NOT PARSE JSON:", e)
        print(raw_text)
        return {"action": "check", "amount": 0}

def to_serializable(x: Any):
    if isinstance(x, (np.uint32, np.uint64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (tuple, list)):
        return [to_serializable(i) for i in x]
    if hasattr(x, 'tolist'):
        try:
            return x.tolist()
        except Exception:
            pass
    return x

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
