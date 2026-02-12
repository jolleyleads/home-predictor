import pandas as pd

REQUIRED_COLS = ["price", "bedrooms", "bathrooms", "square_feet", "zipcode"]

def load_and_clean(raw_path: str, cleaned_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # --- normalize column names (optional helpers) ---
    df.columns = [c.strip().lower() for c in df.columns]

    # If your raw dataset uses different names, map them here:
    rename_map = {
        "saleprice": "price",
        "beds": "bedrooms",
        "baths": "bathrooms",
        "sqft": "square_feet",
        "sq_ft": "square_feet",
        "zip": "zipcode",
    }
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Present columns: {list(df.columns)}")

    # Keep only needed columns for the first version
    df = df[REQUIRED_COLS].copy()

    # Basic cleaning: drop rows with missing target
    df = df.dropna(subset=["price"])

    # Fill missing numeric values with median
    for col in ["bedrooms", "bathrooms", "square_feet"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    # Zipcodes as strings (categorical)
    df["zipcode"] = df["zipcode"].astype(str).str.strip()

    # Remove extreme outliers (simple guardrails)
    df = df[(df["square_feet"] > 200) & (df["square_feet"] < 10000)]
    df = df[(df["price"] > 10000) & (df["price"] < 3000000)]

    df.to_csv(cleaned_path, index=False)
    return df

if __name__ == "__main__":
    load_and_clean("data/raw.csv", "data/cleaned.csv")
    print("Saved cleaned dataset to data/cleaned.csv")
