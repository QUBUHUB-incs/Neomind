import json, os

DATA_PATH = "data/training_data.jsonl"

def append_data(data):
    with open(DATA_PATH, "a") as f:
        for d in data if isinstance(data, list) else [data]:
            json.dump(d, f)
            f.write("\n")

def load_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        return [json.loads(line) for line in f]
