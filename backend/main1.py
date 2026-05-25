from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD TRAINED MODEL
# -------------------------
model = joblib.load("model.pkl")

# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"message": "Backend is running"}

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile):
    try:
        df = pd.read_csv(file.file)

        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())

        if df.shape[0] == 0:
            return {"error": "Dataset contains only invalid values"}

        data = df.values

        # -------------------------
        # Feature Extraction
        # -------------------------
        features = np.nanmean(data, axis=0)
        features = np.nan_to_num(features)

        # Normalize
        max_val = np.max(np.abs(features))
        if max_val != 0:
            features = features / max_val

        # -------------------------
        # MATCH MODEL FEATURES (IMPORTANT 🔥)
        # -------------------------
        expected_features = model.n_features_in_

        if len(features) < expected_features:
            features = np.pad(features, (0, expected_features - len(features)))
        else:
            features = features[:expected_features]

        # -------------------------
        # Prediction
        # -------------------------
        prediction = model.predict([features])[0]

        proba = model.predict_proba([features])[0]
        confidence = float(np.max(proba))

        # -------------------------
        # Dummy brainwave mapping (for UI)
        # -------------------------
        alpha = float(features[0]) if len(features) > 0 else 0
        beta = float(features[1]) if len(features) > 1 else 0
        gamma = float(features[2]) if len(features) > 2 else 0
        theta = float(features[3]) if len(features) > 3 else 0

        return {
            "mental_state": str(prediction),
            "confidence": round(confidence * 100, 2),
            "brain_waves": {
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "theta": theta
            }
        }

    except Exception as e:
        return {"error": str(e)}