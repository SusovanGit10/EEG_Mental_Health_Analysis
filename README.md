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
- CSS

### Backend
- FastAPI
- Python
- TensorFlow
- PyTorch
- Scikit-learn
- XGBoost

### Deployment
- Vercel (Frontend)
- Hugging Face Spaces (Backend)
- Docker

---

## 📂 Project Structure

```bash
EEG_Mental_Health_Analysis/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── *.pkl
│   ├── *.pth
│   └── *.h5
│
├── frontend/
│   └── user/
│       ├── src/
│       ├── public/
│       └── package.json
│
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## ⚡ API Endpoint

### POST `/predict`

Upload EEG CSV data for prediction.

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

---

### Backend Setup

```bash
cd backend

pip install -r requirements.txt

uvicorn main:app --reload
```

Backend runs on:

```bash
http://127.0.0.1:8000
```

---

### Frontend Setup

```bash
cd frontend/user

npm install

npm start
```

Frontend runs on:

```bash
http://localhost:3000
```

---

## 📊 Sample Output

The system provides:

- Final prediction
- Confidence score
- Individual model predictions
- EEG wave analysis
- Comparative performance chart

---

## 🔒 Disclaimer

This project is developed for educational and research purposes only.

It is not intended to replace professional medical diagnosis or treatment.

---

## 👨‍💻 Author

### Susovan Hati

- GitHub: https://github.com/SusovanGit10
- Project: EEG Mental Health Analysis

---

## ⭐ Support

If you like this project:

- Star the repository
- Fork the project
- Share feedback
- Contribute improvements

---

## 📜 License

This project is licensed under the MIT License.