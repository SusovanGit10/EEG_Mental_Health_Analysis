import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Load
df = pd.read_csv("mental-state.csv")

# Clean
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

label_col = df.columns[-1]

X = df.drop(label_col, axis=1)
y = df[label_col]

X = X.select_dtypes(include=[np.number])

# 🔥 Encode labels (important for XGB)
le = LabelEncoder()
y = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🚀 Tuned XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss"
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# Save
joblib.dump({
    "model": model,
    "accuracy": round(accuracy, 2)
}, "xgb_model.pkl")

joblib.dump(le, "xgb_label_encoder.pkl")

print("✅ XGBoost Accuracy:", accuracy)