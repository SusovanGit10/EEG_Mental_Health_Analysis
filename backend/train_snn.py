import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import joblib

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("mental-state.csv")

df = df.select_dtypes(include=[np.number])
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean())

TARGET_COLUMN = df.columns[-1]

X = df.drop(columns=[TARGET_COLUMN]).values
y = df[TARGET_COLUMN].values

# -------------------------
# ENCODE LABELS
# -------------------------
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "snn_label_encoder.pkl")

# -------------------------
# SCALE
# -------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, "snn_scaler.pkl")

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TORCH DATA
# -------------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# -------------------------
# MODEL (IMPROVED)
# -------------------------
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

        # 🔥 ADD THIS
        out = self.fc4(spk3)
        out = torch.relu(out)   # stabilize output

        return out
# -------------------------
# INIT
# -------------------------
input_size = X_train.shape[1]
num_classes = len(np.unique(y))

model = SNNModel(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# TRAIN (FIXED)
# -------------------------
EPOCHS = 80

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for xb, yb in train_loader:
        optimizer.zero_grad()

        outputs = model(xb)
        loss = criterion(outputs, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

# -------------------------
# TEST
# -------------------------
model.eval()

with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)

accuracy = accuracy_score(y_test.numpy(), preds.numpy()) * 100

print(f"\n🔥 SNN Accuracy: {accuracy:.2f}%")

# -------------------------
# SAVE
# -------------------------
torch.save({
    "model_state": model.state_dict(),
    "input_size": input_size,
    "num_classes": num_classes,
    "accuracy": accuracy
}, "snn_model.pth")

print("✅ SNN model saved")