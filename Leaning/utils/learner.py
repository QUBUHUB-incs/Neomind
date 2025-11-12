import random, time, json

METRICS_PATH = "logs/train_metrics.json"

def train_model(model, dataset):
    vectorizer = model["vectorizer"]
    classifier = model["classifier"]

    texts = [item["text"] for item in dataset if "text" in item]
    labels = [item.get("label", "unknown") for item in dataset]

    X = vectorizer.fit_transform(texts)
    classifier.fit(X, labels)
    model["trained"] = True

    logs = [f"Training completed on {len(texts)} samples"]
    with open("logs/train_logs.txt", "a") as f:
        f.write("\n".join(logs) + "\n")

    # Mock metric saving
    metrics = {"accuracy": round(random.uniform(0.7, 0.99), 3), "loss": round(random.uniform(0.01, 0.3), 3)}
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f)
    return logs

def get_metrics():
    try: 
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    except:
        return {"accuracy": 0, "loss": 0, "message": "No training yet"}
