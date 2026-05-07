import json
import os
import joblib
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer

try:
    from sklearn.metrics import root_mean_squared_error
except Exception:  # sklearn<1.4
    root_mean_squared_error = None

from sklearn.metrics import mean_squared_error

MODEL_PATH = os.path.join("models", "model.joblib")
DATA_PATH = os.path.join("data", "cleaned.csv")

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run: python -m src.train")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}. Run: python -m src.preprocess (or python -m src.train)")

    pipe = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["price"])
    y = df["price"]

    # Cross-validated MAE / RMSE (small datasets will be noisy)
    k = min(5, len(df))
    cv = KFold(n_splits=max(2, k), shuffle=True, random_state=42)

    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    rmse_scorer = make_scorer(lambda yt, yp: (root_mean_squared_error(yt, yp) if root_mean_squared_error else mean_squared_error(yt, yp, squared=False)), greater_is_better=False)

    mae_scores = cross_val_score(pipe, X, y, cv=cv, scoring=mae_scorer)
    rmse_scores = cross_val_score(pipe, X, y, cv=cv, scoring=rmse_scorer)

    out = {
        "cv_splits": int(cv.get_n_splits()),
        "mae_mean": float(-mae_scores.mean()),
        "mae_std": float(mae_scores.std()),
        "rmse_mean": float(-rmse_scores.mean()),
        "rmse_std": float(rmse_scores.std()),
        "rows": int(len(df)),
    }

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
