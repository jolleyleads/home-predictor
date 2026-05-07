import csv
import json
import os
import joblib
<<<<<<< HEAD
import pandas as pd

from src.preprocess import load_and_clean
=======
>>>>>>> a1d9516b99133ebcc85e43a52c950cc031947c72

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
<<<<<<< HEAD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_CSV = os.path.join("data", "RAW.csv")
CLEANED_CSV = os.path.join("data", "cleaned.csv")
=======
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_CSV = os.path.join("data", "raw.csv")
>>>>>>> a1d9516b99133ebcc85e43a52c950cc031947c72
MODEL_OUT = os.path.join("models", "model.joblib")
META_OUT = os.path.join("models", "metadata.json")

FEATURES_NUM = ["bedrooms", "bathrooms", "square_feet"]
FEATURES_CAT = ["zipcode"]
TARGET = "price"


def load_csv(path: str):
<<<<<<< HEAD
    """Load cleaned CSV and return X as a pandas DataFrame (recommended for ColumnTransformer)."""
    df = pd.read_csv(path)

    # enforce dtypes
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    for c in FEATURES_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[FEATURES_CAT[0]] = df[FEATURES_CAT[0]].astype(str).str.strip()

    df = df.dropna(subset=[TARGET])
    X = df[FEATURES_NUM + FEATURES_CAT].copy()
    y = df[TARGET].astype(float).tolist()
    return X, y, int(len(df))

=======
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
>>>>>>> a1d9516b99133ebcc85e43a52c950cc031947c72


def main():
    os.makedirs("models", exist_ok=True)

<<<<<<< HEAD
        # Ensure cleaned dataset exists (cleaning is idempotent)
    if not os.path.exists(CLEANED_CSV):
        os.makedirs("data", exist_ok=True)
        load_and_clean(RAW_CSV, CLEANED_CSV)

    X, y, nrows = load_csv(CLEANED_CSV)
=======
    X, y, nrows = load_csv(RAW_CSV)
>>>>>>> a1d9516b99133ebcc85e43a52c950cc031947c72
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
<<<<<<< HEAD
            ("num", StandardScaler(), FEATURES_NUM),
=======
            ("num", "passthrough", FEATURES_NUM),
>>>>>>> a1d9516b99133ebcc85e43a52c950cc031947c72
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

<<<<<<< HEAD
    model = Ridge(alpha=1.0, random_state=42)
=======
    model = SGDRegressor(
        random_state=42,
        max_iter=2000,
        tol=1e-3
    )
>>>>>>> a1d9516b99133ebcc85e43a52c950cc031947c72

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
