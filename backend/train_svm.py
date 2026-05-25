import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

# 🔥 SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = SVC(probability=True)
model.fit(X_train, y_train)

# 🔥 ACCURACY
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

# 🔥 SAVE
joblib.dump({
    "model": model,
    "accuracy": round(accuracy, 2)
}, "svm_model.pkl")

print("✅ SVM Accuracy:", accuracy)