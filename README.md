---
title: EEG Mental Health Analysis
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🧠 EEG Mental Health Analysis

AI-powered EEG Mental Health Analysis System built using React, FastAPI, TensorFlow, PyTorch, Scikit-learn, and XGBoost.

---

## 🚀 Live Demo

### 🌐 Frontend
https://eeg-mental-health-analysis.vercel.app

### 🤗 Backend API
https://susovanhuggingface10-eeg-mental-health-analysis.hf.space

---

## 📌 Features

- EEG signal-based mental health prediction
- Multiple AI/ML model integration
- Beautiful React dashboard UI
- FastAPI backend API
- Real-time prediction system
- Comparative model analysis
- Interactive charts and visualization
- Dockerized deployment

---

## 🧠 Models Used

| Model | Framework |
|-------|------------|
| Random Forest | Scikit-learn |
| SVM | Scikit-learn |
| CNN | TensorFlow / Keras |
| XGBoost | XGBoost |
| SNN | PyTorch |

---

## 🛠️ Tech Stack

### Frontend
- React.js
- Axios
- Recharts

### Backend
- FastAPI
- TensorFlow
- PyTorch
- Scikit-learn
- XGBoost

### Deployment
- Vercel
- Hugging Face Spaces
- Docker

---

## ⚡ API Endpoint

POST `/predict`

Example:

```python
import requests

url = "https://susovanhuggingface10-eeg-mental-health-analysis.hf.space/predict"

files = {
    "file": open("sample.csv", "rb")
}

response = requests.post(url, files=files)

print(response.json())
```

---

## 💻 Local Setup

### Clone Repository

```bash
git clone https://github.com/SusovanGit10/EEG_Mental_Health_Analysis.git
cd EEG_Mental_Health_Analysis
```

### Backend Setup

```bash
cd backend

pip install -r requirements.txt

uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend/user

npm install

npm start
```

---

## 🔒 Disclaimer

This project is for educational and research purposes only.

---

## 👨‍💻 Author

Susovan Hati

GitHub:
https://github.com/SusovanGit10

---

## 📜 License

MIT License