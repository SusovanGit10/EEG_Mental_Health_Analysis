from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ✅ SNN imports
import torch
import torch.nn as nn
import snntorch as snn

app = FastAPI()

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# LOAD MODELS
# -------------------------
rf_data = joblib.load("rf_model.pkl")
svm_data = joblib.load("svm_model.pkl")
xgb_data = joblib.load("xgb_model.pkl")

rf_model = rf_data["model"]
rf_accuracy = rf_data["accuracy"]

svm_model = svm_data["model"]
svm_accuracy = svm_data["accuracy"]

xgb_model = xgb_data["model"]
xgb_accuracy = xgb_data["accuracy"]

xgb_label_encoder = joblib.load("xgb_label_encoder.pkl")

cnn_model = load_model("cnn_model.h5")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

cnn_accuracy = joblib.load("cnn_meta.pkl")["accuracy"]

# -------------------------
# LOAD SNN
# -------------------------
snn_checkpoint = torch.load("snn_model.pth", map_location="cpu")

snn_input_size = snn_checkpoint["input_size"]
snn_num_classes = snn_checkpoint["num_classes"]
snn_accuracy = snn_checkpoint["accuracy"]

class SNNModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(256, 128)
        self.lif2 = snn.Leaky(beta=0.9)

        self.fc3 = nn.Linear(128, 64)
        self.lif3 = snn.Leaky(beta=0.9)

        self.dropout = nn.Dropout(0.3)

        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk1, mem1 = self.lif1(self.fc1(x), mem1)
        spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
        spk2 = self.dropout(spk2)

        spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
        out = self.fc4(spk3)

        return out

snn_model = SNNModel(snn_input_size, snn_num_classes)
snn_model.load_state_dict(snn_checkpoint["model_state"],strict=False)
snn_model.eval()

snn_label_encoder = joblib.load("snn_label_encoder.pkl")
snn_scaler = joblib.load("snn_scaler.pkl")

# -------------------------
# EEG BAND EXTRACTION
# -------------------------
def extract_brain_waves(features):
    n = len(features)
    q = n // 4

    return {
        "alpha": float(np.mean(features[:q])),
        "beta": float(np.mean(features[q:2*q])),
        "gamma": float(np.mean(features[2*q:3*q])),
        "theta": float(np.mean(features[3*q:]))
    }

# -------------------------
# ANALYSIS
# -------------------------
def analyze(waves, prediction):
    alpha = waves["alpha"]
    beta = waves["beta"]
    gamma = waves["gamma"]
    theta = waves["theta"]

    return {
        "condition": prediction,
        "cognitive_activity": "High" if gamma > alpha else "Low",
        "focus_score": round(float(gamma / (alpha + 1e-6)), 2),
        "stress_index": round(float(beta / (theta + 1e-6)), 2),
    }

# -------------------------
# SPIKE ENCODING
# -------------------------
def spike_encoding(features):
    threshold = np.mean(features)
    return (features > threshold).astype(float)

# -------------------------
# HOME
# -------------------------
@app.get("/")
def home():
    return {"message": "Backend running 🚀"}

# -------------------------
# PREDICT
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile):
    try:
        df = pd.read_csv(file.file)

        # CLEAN
        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())

        data = df.values

        # FEATURE VECTOR
        features = np.mean(data, axis=0)
        features = np.nan_to_num(features)

        # FIX SIZE
        expected = scaler.n_features_in_
        if len(features) > expected:
            features = features[:expected]
        elif len(features) < expected:
            features = np.pad(features, (0, expected - len(features)))

        brain_waves = extract_brain_waves(features)

        def prepare_input(model, features):
            n = model.n_features_in_
            if len(features) > n:
                return features[:n]
            elif len(features) < n:
                return np.pad(features, (0, n - len(features)))
            return features

        rf_input = prepare_input(rf_model, features)
        svm_input = prepare_input(svm_model, features)
        xgb_input = prepare_input(xgb_model, features)

        # RF
        rf_df = pd.DataFrame([rf_input], columns=rf_model.feature_names_in_)
        rf_pred = rf_model.predict(rf_df)[0]
        rf_conf = float(np.max(rf_model.predict_proba(rf_df)[0]) * 100)

        # SVM
        svm_df = pd.DataFrame([svm_input], columns=svm_model.feature_names_in_)
        svm_pred = svm_model.predict(svm_df)[0]
        svm_conf = float(np.max(svm_model.predict_proba(svm_df)[0]) * 100)

        # XGB

        xgb_df = pd.DataFrame([xgb_input], columns=xgb_model.feature_names_in_)

        xgb_pred_raw = xgb_model.predict(xgb_df)[0]

        # 🔥 THIS LINE WAS MISSING
        xgb_pred = xgb_label_encoder.inverse_transform([xgb_pred_raw])[0]

        xgb_conf = float(np.max(xgb_model.predict_proba(xgb_df)[0]) * 100)

        # CNN
        features_df = pd.DataFrame([features], columns=scaler.feature_names_in_)
        features_scaled = scaler.transform(features_df)
        cnn_input = features_scaled.reshape(1, features_scaled.shape[1], 1)
        cnn_pred_raw = cnn_model.predict(cnn_input)[0]
        cnn_class = label_encoder.inverse_transform([np.argmax(cnn_pred_raw)])[0]
        cnn_conf = float(np.max(cnn_pred_raw) * 100)

        # -------------------------
        # SNN
        # -------------------------
# -------------------------
# SNN (FIXED)
# -------------------------
        try:
            snn_scaled = snn_scaler.transform([features])[0]

            snn_input = torch.tensor(snn_scaled, dtype=torch.float32)

            with torch.no_grad():
                snn_output = snn_model(snn_input)
                probs = torch.softmax(snn_output, dim=0)

                snn_pred_idx = torch.argmax(probs).item()
                snn_pred = snn_label_encoder.inverse_transform([snn_pred_idx])[0]
                snn_conf = float(torch.max(probs).item() * 100)

        except Exception as e:
            print("SNN ERROR:", e)
            snn_pred = "N/A"
            snn_conf = 0
        # ANALYSIS
        rf_analysis = analyze(brain_waves, str(rf_pred))
        svm_analysis = analyze(brain_waves, str(svm_pred))
        xgb_analysis = analyze(brain_waves, str(xgb_pred))
        cnn_analysis = analyze(brain_waves, str(cnn_class))
        snn_analysis = analyze(brain_waves, str(snn_pred))

        # ENSEMBLE
        predictions = [
            ("RF", rf_pred, rf_conf, rf_accuracy),
            ("SVM", svm_pred, svm_conf, svm_accuracy),
            ("XGB", xgb_pred, xgb_conf, xgb_accuracy),
            ("CNN", cnn_class, cnn_conf, cnn_accuracy),
            ("SNN", snn_pred, snn_conf, snn_accuracy),
        ]

        best = max(predictions, key=lambda x: (x[2] * 0.6 + x[3] * 0.4))
        best_score = best[2] * 0.6 + best[3] * 0.4

        return {
            "random_forest": {
                "prediction": str(rf_pred),
                "confidence": round(rf_conf, 2),
                "accuracy": rf_accuracy,
                "brain_waves": brain_waves,
                "analysis": rf_analysis
            },
            "svm": {
                "prediction": str(svm_pred),
                "confidence": round(svm_conf, 2),
                "accuracy": svm_accuracy,
                "brain_waves": brain_waves,
                "analysis": svm_analysis
            },
            "xgboost": {
                "prediction": str(xgb_pred),
                "confidence": round(xgb_conf, 2),
                "accuracy": xgb_accuracy,
                "brain_waves": brain_waves,
                "analysis": xgb_analysis
            },
            "cnn": {
                "prediction": str(cnn_class),
                "confidence": round(cnn_conf, 2),
                "accuracy": cnn_accuracy,
                "brain_waves": brain_waves,
                "analysis": cnn_analysis
            },
            "snn": {
                "prediction": str(snn_pred),
                "confidence": round(snn_conf, 2),
                "accuracy": snn_accuracy,
                "brain_waves": brain_waves,
                "analysis": snn_analysis
            },
            "final_decision": {
                "model": best[0],
                "prediction": str(best[1]),
                "confidence": round(best[2], 2),
                "score": round(best_score, 2)
            }
        }

    except Exception as e:
        return {"error": str(e)}