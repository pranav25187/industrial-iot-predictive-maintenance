from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# -----------------------------
# Resolve absolute paths safely
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lstm_model.h5")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")

# -----------------------------
# Load model and scaler
# -----------------------------
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Model or scaler loading failed: {e}")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Industrial IoT Predictive Maintenance API",
    description="Predict Remaining Useful Life (RUL) of turbofan engines",
    version="1.0"
)

# -----------------------------
# Input schema
# -----------------------------
class SensorInput(BaseModel):
    engine_id: int
    sensor_readings: list  # expected shape: [50][18]

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "API is running"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_rul(data: SensorInput):
    try:
        print("=== DEBUG: Request received ===")

        sensor_array = np.array(data.sensor_readings, dtype=np.float32)
        print("DEBUG: Raw input shape:", sensor_array.shape)

        if sensor_array.shape != (50, 17):
            raise ValueError(f"Expected (50,17), got {sensor_array.shape}")

        model_input = sensor_array.reshape(1, 50, 17)
        print("DEBUG: Model input shape:", model_input.shape)

        rul_pred = float(model.predict(model_input, verbose=0)[0][0])
        print("DEBUG: Prediction successful:", rul_pred)

        if rul_pred < 50:
            status = "Critical"
            recommendation = "Immediate maintenance required"
        elif rul_pred < 100:
            status = "Warning"
            recommendation = "Schedule maintenance soon"
        else:
            status = "Healthy"
            recommendation = "No immediate action required"

        return {
            "engine_id": data.engine_id,
            "predicted_rul": round(rul_pred, 2),
            "status": status,
            "recommendation": recommendation
        }

    except Exception as e:
        print("🔥 DEBUG ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

