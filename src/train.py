import csv
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_CSV = os.path.join("data", "raw.csv")
MODEL_OUT = os.path.join("models", "model.joblib")
META_OUT = os.path.join("models", "metadata.json")

FEATURES_NUM = ["bedrooms", "bathrooms", "square_feet"]
FEATURES_CAT = ["zipcode"]
TARGET = "price"


def load_csv(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    X, y = [], []
    for r in rows:
        y.append(float(r[TARGET]))
        X.append({
            "bedrooms": float(r["bedrooms"]),
            "bathrooms": float(r["bathrooms"]),
            "square_feet": float(r["square_feet"]),
            "zipcode": str(r["zipcode"]).strip(),
        })
    return X, y, len(rows)


def main():
    os.makedirs("models", exist_ok=True)

    X, y, nrows = load_csv(RAW_CSV)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = SGDRegressor(
        random_state=42,
        max_iter=2000,
        tol=1e-3
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    joblib.dump(pipe, MODEL_OUT)

    meta = {
        "features_num": FEATURES_NUM,
        "features_cat": FEATURES_CAT,
        "target": TARGET,
        "rows": nrows,
        "mae": float(mae),
        "rmse": float(rmse),
    }
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved:", MODEL_OUT)
    print("✅ Saved:", META_OUT)
    print("MAE:", mae)
    print("RMSE:", rmse)


if __name__ == "__main__":
    main()
