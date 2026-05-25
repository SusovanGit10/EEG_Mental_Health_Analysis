import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.utils import to_categorical

# Load
df = pd.read_csv("mental-state.csv")

# Clean
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

label_col = df.columns[-1]

X = df.drop(label_col, axis=1)
y = df[label_col]

X = X.select_dtypes(include=[np.number])

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 🔥 SPLIT BEFORE RESHAPE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Model
model = Sequential([
    Conv1D(32, 3, activation='relu', input_shape=(X.shape[1], 1)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(y_train_cat.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train_cat, epochs=5, batch_size=32)

# 🔥 ACCURACY
pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

accuracy = accuracy_score(y_test, pred_classes) * 100

# Save
model.save("cnn_model.h5")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save accuracy separately
joblib.dump({"accuracy": round(accuracy, 2)}, "cnn_meta.pkl")

print("✅ CNN Accuracy:", accuracy)