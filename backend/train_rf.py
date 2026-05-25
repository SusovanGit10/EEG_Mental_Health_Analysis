import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load
df = pd.read_csv("mental-state.csv")

# Clean
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

# Label detect
label_col = df.columns[-1]

X = df.drop(label_col, axis=1)
y = df[label_col]

X = X.select_dtypes(include=[np.number])

# 🔥 SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 🔥 ACCURACY
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# 🔥 SAVE PROPERLY
joblib.dump({
    "model": model,
    "accuracy": round(accuracy, 2)
}, "rf_model.pkl")

print("✅ RF Accuracy:", accuracy)