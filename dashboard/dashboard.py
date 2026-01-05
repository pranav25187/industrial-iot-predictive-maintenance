import streamlit as st
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
import joblib
import os

st.set_page_config(
    page_title="Industrial IoT Predictive Maintenance",
    layout="wide"
)

@st.cache_resource
def load_assets():
    model_path = os.path.join("models", "lstm_model.h5")
    scaler_path = os.path.join("models", "scaler.pkl")
    
    # Load model without compiling to bypass version-specific LSTM arguments
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Re-compile manually with basics to ensure it's ready for predict()
    model.compile(optimizer='adam', loss='mse')
    
    scaler = joblib.load(scaler_path)
    return model, scaler
try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model/scaler: {e}")
    st.stop()

st.title("🛠 Industrial IoT Predictive Maintenance Dashboard")
st.markdown("Predict Remaining Useful Life (RUL) using sensor time-series data")

st.sidebar.header("Simulation Settings")
engine_id = st.sidebar.number_input(
    "Engine ID",
    min_value=1,
    max_value=999,
    value=1
)

st.subheader("Sensor Input (Last 50 Cycles)")
if st.button("Generate Sample Sensor Data"):
    sensor_data = np.random.rand(50, 17).tolist()
    st.session_state["sensor_data"] = sensor_data

if "sensor_data" not in st.session_state:
    st.warning("Click 'Generate Sample Sensor Data' to proceed")
    st.stop()

sensor_data = np.array(st.session_state["sensor_data"])

st.subheader("Sensor Trends")
fig = go.Figure()
for i in range(3):
    fig.add_trace(
        go.Scatter(
            y=sensor_data[:, i],
            mode="lines",
            name=f"Sensor {i+1}"
        )
    )
fig.update_layout(xaxis_title="Cycle", yaxis_title="Normalized Value", height=350)
st.plotly_chart(fig, use_container_width=True)

if st.button("🔍 Predict RUL"):
    with st.spinner("Processing local inference..."):
        input_data = sensor_data.reshape(1, 50, 17)
        prediction = model.predict(input_data)
        predicted_rul = int(prediction[0][0])
        
        if predicted_rul < 50:
            status = "Danger"
            recommendation = "Immediate Maintenance Required"
        elif predicted_rul < 100:
            status = "Warning"
            recommendation = "Schedule Maintenance Soon"
        else:
            status = "Healthy"
            recommendation = "No Action Needed"

    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted RUL (cycles)", f"{predicted_rul}")
    col2.metric("Health Status", status)
    col3.metric("Recommendation", recommendation)

    st.subheader("Remaining Useful Life Indicator")
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=predicted_rul,
            gauge={
                "axis": {"range": [0, 200]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "red"},
                    {"range": [50, 100], "color": "orange"},
                    {"range": [100, 200], "color": "green"},
                ],
            },
        )
    )
    gauge.update_layout(height=350)
    st.plotly_chart(gauge, use_container_width=True)