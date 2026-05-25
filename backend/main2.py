from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
rf_model = joblib.load("model.pkl")
svm_model = joblib.load("svm_model.pkl")

@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        df = pd.read_csv(file.file)

        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())

        if df.shape[0] == 0:
            return {"error": "Invalid dataset"}

        data = df.values

        # Feature extraction
        features = np.nanmean(data, axis=0)
        features = np.nan_to_num(features)

        max_val = np.max(np.abs(features))
        if max_val != 0:
            features = features / max_val

        # Match RF
        rf_size = rf_model.n_features_in_
        f_rf = np.pad(features, (0, max(0, rf_size - len(features))))[:rf_size]

        # Match SVM
        svm_size = svm_model.n_features_in_
        f_svm = np.pad(features, (0, max(0, svm_size - len(features))))[:svm_size]

        # Predictions
        rf_pred = rf_model.predict([f_rf])[0]
        rf_conf = float(np.max(rf_model.predict_proba([f_rf])[0]))

        svm_pred = svm_model.predict([f_svm])[0]
        svm_conf = float(np.max(svm_model.predict_proba([f_svm])[0]))

        # Brain waves
        def get_waves(f):
            return {
                "alpha": float(f[0]) if len(f) > 0 else 0,
                "beta": float(f[1]) if len(f) > 1 else 0,
                "gamma": float(f[2]) if len(f) > 2 else 0,
                "theta": float(f[3]) if len(f) > 3 else 0,
            }

        # Extra analysis
        def analyze(w):
            alpha, beta, gamma, theta = w["alpha"], w["beta"], w["gamma"], w["theta"]

            condition = "Relaxed" if alpha > beta else "Stressed"
            cognitive = "High" if gamma > 0.5 else "Normal"
            focus = beta - theta
            stress = beta / (alpha + 1e-6)

            return {
                "condition": condition,
                "cognitive_activity": cognitive,
                "focus_score": float(focus),
                "stress_index": float(stress)
            }

        rf_waves = get_waves(f_rf)
        svm_waves = get_waves(f_svm)

        return {
            "random_forest": {
                "prediction": str(rf_pred),
                "confidence": round(rf_conf * 100, 2),
                "brain_waves": rf_waves,
                "analysis": analyze(rf_waves)
            },
            "svm": {
                "prediction": str(svm_pred),
                "confidence": round(svm_conf * 100, 2),
                "brain_waves": svm_waves,
                "analysis": analyze(svm_waves)
            }
        }

    except Exception as e:
        return {"error": str(e)}