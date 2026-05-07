import argparse
import json
import os
import joblib

MODEL_PATH = os.path.join("models", "model.joblib")

def parse_args():
    p = argparse.ArgumentParser(description="Predict house price from features.")
    p.add_argument("--bedrooms", type=float, required=True)
    p.add_argument("--bathrooms", type=float, required=True)
    p.add_argument("--square_feet", type=float, required=True)
    p.add_argument("--zipcode", type=str, required=True)
    return p.parse_args()

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it with: python -m src.train")

    model = joblib.load(MODEL_PATH)
    X = [{
        "bedrooms": float(parse_args().bedrooms),
        "bathrooms": float(parse_args().bathrooms),
        "square_feet": float(parse_args().square_feet),
        "zipcode": str(parse_args().zipcode).strip()
    }]
    pred = float(model.predict(X)[0])
    print(json.dumps({"predicted_price": pred}, indent=2))

if __name__ == "__main__":
    main()
