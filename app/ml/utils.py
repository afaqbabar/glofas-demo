import json, os
MODELS_DIR = os.getenv("MODELS_DIR", "app/models")

def load_thresholds():
    path = os.path.join(MODELS_DIR, "thresholds.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)