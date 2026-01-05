import streamlit as st
import numpy as np
import requests
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000/predict"

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Industrial IoT Predictive Maintenance",
    layout="wide"
)

st.title("🛠 Industrial IoT Predictive Maintenance Dashboard")
st.markdown("Predict Remaining Useful Life (RUL) using sensor time-series data")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Simulation Settings")

engine_id = st.sidebar.number_input(
    "Engine ID",
    min_value=1,
    max_value=999,
    value=1
)

# -----------------------------
# Generate dummy sensor data
# -----------------------------
st.subheader("Sensor Input (Last 50 Cycles)")

if st.button("Generate Sample Sensor Data"):
    sensor_data = np.random.rand(50, 17).tolist()
    st.session_state["sensor_data"] = sensor_data

if "sensor_data" not in st.session_state:
    st.warning("Click 'Generate Sample Sensor Data' to proceed")
    st.stop()

sensor_data = np.array(st.session_state["sensor_data"])

# -----------------------------
# Sensor trend visualization
# -----------------------------
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

fig.update_layout(
    xaxis_title="Cycle",
    yaxis_title="Normalized Value",
    height=350
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Predict button
# -----------------------------
if st.button("🔍 Predict RUL"):

    payload = {
        "engine_id": engine_id,
        "sensor_readings": sensor_data.tolist()
    }

    with st.spinner("Calling predictive maintenance API..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code != 200:
        st.error("API Error — check FastAPI server")
        st.stop()

    result = response.json()

    # -----------------------------
    # Results
    # -----------------------------
    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted RUL (cycles)",
        f"{result['predicted_rul']}"
    )

    col2.metric(
        "Health Status",
        result["status"]
    )

    col3.metric(
        "Recommendation",
        result["recommendation"]
    )

    # -----------------------------
    # RUL Gauge
    # -----------------------------
    st.subheader("Remaining Useful Life Indicator")

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result["predicted_rul"],
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
