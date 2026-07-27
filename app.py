import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------- Page Config --------------------
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# -------------------- Load Model & Data --------------------
df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

# -------------------- Title --------------------
st.title("💻 Laptop Price Predictor")
st.markdown("Fill in the laptop specifications below to get an estimated price.")
st.divider()

# -------------------- Input Layout --------------------
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox("Brand", sorted(df['Company'].unique()))
    type_name = st.selectbox("Type", sorted(df['TypeName'].unique()))
    ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
    weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
    ips = st.selectbox("IPS Display", ["No", "Yes"])

with col2:
    screen_size = st.number_input("Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    resolution = st.selectbox(
        "Screen Resolution",
        ["1920x1080", "1366x768", "1600x900", "3840x2160",
         "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"]
    )
    cpu = st.selectbox("CPU Brand", sorted(df['Cpu brand'].unique()))
    hdd = st.selectbox("HDD (GB)", sorted(df['HDD'].unique()))
    ssd = st.selectbox("SSD (GB)", sorted(df['SSD'].unique()))

col3, col4 = st.columns(2)
with col3:
    gpu = st.selectbox("GPU Brand", sorted(df['Gpu brand'].unique()))
with col4:
    os = st.selectbox("Operating System", sorted(df['os'].unique()))

st.divider()

# -------------------- Predict --------------------
if st.button("Predict Price", type="primary", use_container_width=True):

    # Convert Yes/No to binary
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Calculate ppi from resolution + screen size
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size

    # Build input DataFrame matching training columns
    query = pd.DataFrame([{
        'Company': company,
        'TypeName': type_name,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen_val,
        'Ips': ips_val,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os
    }])

    # Predict (model trained on log(Price), so invert with exp)
    predicted_log_price = pipe.predict(query)[0]
    predicted_price = np.exp(predicted_log_price)

    st.success(f"### Estimated Price: ₹{int(predicted_price):,}")
