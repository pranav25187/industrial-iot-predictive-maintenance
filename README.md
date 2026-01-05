# 🛠 Industrial IoT Predictive Maintenance Dashboard (MLOps)

An **end-to-end Industrial IoT Predictive Maintenance system** that predicts the **Remaining Useful Life (RUL)** of turbofan engines using **LSTM-based time-series modeling**, served via **FastAPI**, tracked with **MLflow**, and visualized through an interactive **Streamlit dashboard**.

🔗 **Live Demo**:  
👉 https://pranav25187-industrial-iot-predictive-dashboarddashboard-95zkcn.streamlit.app/

---

## 🚀 Key Features

- 📈 LSTM-based Remaining Useful Life (RUL) prediction  
- ⚙️ Industrial sensor time-series processing  
- 🔁 MLflow experiment tracking & model versioning  
- 🌐 FastAPI backend for real-time inference  
- 📊 Streamlit dashboard for interactive visualization  
- 🧠 Health classification (Healthy / Warning / Critical)  
- 🏭 Maintenance recommendation system  

---

## 🧠 Tech Stack

- **Programming Language**: Python  
- **Machine Learning / Deep Learning**: NumPy, Pandas, Scikit-learn, TensorFlow (LSTM)  
- **MLOps**: MLflow  
- **Backend API**: FastAPI  
- **Frontend Dashboard**: Streamlit, Plotly  
- **Model Format**: `.h5`  

---

## 📂 Project Structure

```

industrial-iot-predictive-maintenance/
│
├── api/
│   └── main.py               # FastAPI inference service
│
├── dashboard/
│   └── dashboard.py          # Streamlit dashboard
│
├── models/
│   └── lstm_model.h5         # Trained LSTM model
│
├── notebooks/
│   └── model_training.ipynb  # Model training & MLflow logging
│
├── requirements.txt
└── README.md

````

---

## 📊 Dashboard Preview

- Sensor trend visualization
- <img width="1916" height="836" alt="image" src="https://github.com/user-attachments/assets/37d88743-620a-456e-8107-d7b61c0d8d4e" />

- RUL prediction & health status
- <img width="1567" height="619" alt="image" src="https://github.com/user-attachments/assets/160c6373-ddee-45ec-85df-4b2b2b060418" />

- Gauge indicator  
<img width="1494" height="350" alt="newplot" src="https://github.com/user-attachments/assets/642cb83b-4e7c-4698-922d-4d6c3197a7ea" />


```markdown
![Dashboard Overview](images/dashboard_overview.png)
![RUL Prediction](images/rul_prediction.png)
````

---

## ⚙️ Local Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/industrial-iot-predictive-maintenance.git
cd industrial-iot-predictive-maintenance
```

### 2️⃣ Create & Activate Virtual Environment

```bash
python -m venv venv
```

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project Locally

### 🔹 Start FastAPI Backend

```bash
uvicorn api.main:app --reload
```

Open API docs:

```
http://127.0.0.1:8000/docs
```

---

### 🔹 Start Streamlit Dashboard

```bash
python -m streamlit run dashboard/dashboard.py
```

Open dashboard:

```
http://localhost:8501
```

---

## 📈 How the System Works

1. Sensor time-series data (last 50 cycles × 17 sensors) is provided
2. LSTM model predicts Remaining Useful Life (RUL)
3. FastAPI serves predictions via REST API
4. Streamlit dashboard visualizes:

   * Sensor trends
   * Predicted RUL
   * Health status
   * Maintenance recommendations

---

## 🧪 MLflow Experiment Tracking

MLflow is used to:

* Track experiments
* Log parameters & metrics
* Store trained models

Run MLflow UI locally:

```bash
mlflow ui
```

Open:

```
http://localhost:5000
```

---

## 🧠 Health Classification Logic

| Predicted RUL | Status   | Recommendation                 |
| ------------- | -------- | ------------------------------ |
| < 50          | Critical | Immediate maintenance required |
| 50 – 100      | Warning  | Schedule maintenance soon      |
| > 100         | Healthy  | No immediate action required   |

---

## 🎯 Learning Outcomes

* Time-series modeling using LSTM
* End-to-end MLOps workflow
* Model serving with FastAPI
* Experiment tracking with MLflow
* Frontend–backend integration
* Real-world predictive maintenance use case

---

## 📌 Author

**Pranav**
Final Year Computer Engineering Student
Aspiring **ML / Data / MLOps Engineer**

---

## 📜 License

This project is intended for **educational and demonstration purposes**.

