import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the bundled pipeline + dataframe
model_data = joblib.load('model_data.pkl')
pipe = model_data['pipeline']
df = model_data['df']

st.title("💻 Laptop Price Predictor")

# Build dropdowns from the training data's actual categories
company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.number_input('Screen Size (inches)', min_value=10.0, max_value=18.0, value=15.6)
resolution = st.selectbox('Screen Resolution', 
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Compute PPI from resolution and screen size — same formula used in training
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2) + (Y_res**2)) ** 0.5 / screen_size

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

    predicted_log_price = pipe.predict(query)
    predicted_price = np.exp(predicted_log_price[0])  # reverse the log transform
    st.success(f"Estimated Price: ₹{int(predicted_price):,}")