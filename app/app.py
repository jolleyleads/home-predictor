from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import json

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")

app = Flask(__name__)
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m src.train"
        )
    return joblib.load(MODEL_PATH)

model = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict():
    global model
    payload = request.get_json(force=True)

    # Expecting:
    # {
    #   "bedrooms": 3,
    #   "bathrooms": 2,
    #   "square_feet": 1700,
    #   "zipcode": "23704"
    # }

    required = ["bedrooms", "bathrooms", "square_feet", "zipcode"]
    missing = [k for k in required if k not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    X = pd.DataFrame([{
        "bedrooms": float(payload["bedrooms"]),
        "bathrooms": float(payload["bathrooms"]),
        "square_feet": float(payload["square_feet"]),
        "zipcode": str(payload["zipcode"]).strip(),
    }])
    if model is None:
        model = load_model()

    pred = float(model.predict(X)[0])
    return jsonify({"predicted_price": pred})


@app.get("/metadata")
def metadata():
    meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "metadata.json")
    if not os.path.exists(meta_path):
        return jsonify({"error": "metadata.json not found. Train the model first."}), 404
    with open(meta_path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
