"""
   Helpers for saving and loading pickle and JSON files.
"""

import json
import os
import pickle


def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[io] Saved → {path}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(data, path, indent=2):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
    print(f"[io] Saved → {path}")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
