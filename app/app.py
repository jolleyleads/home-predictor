from flask import Flask, request, jsonify
import os
import joblib

# --- MODEL PATH (robust for local + Render) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.joblib")
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)

model = None

def load_model():
    global model
    if model is not None:
        return model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model

HOME_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Housing Price Predictor</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 16px; }
    label { display:block; margin-top: 10px; }
    input { width: 100%; padding: 10px; font-size: 16px; margin-top: 6px; }
    button { margin-top: 14px; padding: 10px 14px; font-size: 16px; cursor: pointer; }
    .out { margin-top: 16px; padding: 12px; background: #f6f6f6; border-radius: 8px; white-space: pre-wrap; }
    .small { color:#666; font-size: 13px; margin-top: 10px; }
    code { background:#eee; padding:2px 4px; border-radius: 4px; }
  </style>
</head>
<body>
  <h2>🏠 Housing Price Predictor</h2>
  <div class="card">
    <label>Bedrooms <input id="bedrooms" type="number" step="1" value="3"></label>
    <label>Bathrooms <input id="bathrooms" type="number" step="0.5" value="2"></label>
    <label>Square Feet <input id="square_feet" type="number" step="1" value="1600"></label>
    <label>Zipcode <input id="zipcode" type="text" value="23704"></label>
    <button onclick="predict()">Predict</button>
    <div class="out" id="out">Result will appear here.</div>
    <div class="small">
      API endpoints: <code>/health</code>, <code>/predict</code>, <code>/routes</code><br/>
      Model path: <code>models/model.joblib</code> (used if present; otherwise fallback)
    </div>
  </div>

<script>
async function predict(){
  const payload = {
    bedrooms: Number(document.getElementById("bedrooms").value),
    bathrooms: Number(document.getElementById("bathrooms").value),
    square_feet: Number(document.getElementById("square_feet").value),
    zipcode: String(document.getElementById("zipcode").value || "")
  };
  const out = document.getElementById("out");
  out.textContent = "Calling /predict...";
  try{
    const r = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    out.textContent = JSON.stringify(data, null, 2);
  }catch(e){
    out.textContent = "Error: " + e.toString();
  }
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return HOME_HTML

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/routes")
def routes():
    return jsonify({
        "running_file": os.path.abspath(__file__),
        "cwd": os.getcwd(),
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "routes": sorted([str(r) for r in app.url_map.iter_rules()])
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    # Parse inputs
    try:
        bedrooms = float(data.get("bedrooms", 0))
        bathrooms = float(data.get("bathrooms", 0))
        square_feet = float(data.get("square_feet", 0))
        zipcode = str(data.get("zipcode", ""))
    except Exception:
        return jsonify({"error": "Bad input types"}), 400

    # Prefer real model if present
    model = load_model()
    if model is not None:
        try:
            df = pd.DataFrame([{
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "square_feet": square_feet,
                "zipcode": zipcode
            }])
            pred = float(model.predict(df)[0])
            return jsonify({"prediction": pred, "mode": "model"})
        except Exception as e:
            # Fall back if model expects different columns/etc
            pass

    # Fallback prediction (never fails)
    pred = 50000 + (square_feet * 150) + (bedrooms * 20000) + (bathrooms * 15000)
    return jsonify({"prediction": float(pred), "mode": "fallback"})

if __name__ == "__main__":
    # use_reloader=False prevents “double server” confusion
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True, use_reloader=False)
