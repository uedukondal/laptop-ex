
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("💻 Laptop Price Predictor")

# Load saved model and data
df = pickle.load(open("df.pkl", "rb"))
pipe = pickle.load(open("pipe.pkl", "rb"))
laptop = pd.read_csv("sample_laptop.csv")

# User inputs
brand = st.selectbox("Select Laptop Brand", df['brand'].unique())
cpu_brand = st.selectbox("CPU Brand", df['cpu_brand'].unique())
cpu_gen = st.slider("CPU Generation", 1, 13, 11)
cpu_cores = st.slider("CPU Cores", 2, 16, 6)
cpu_threads = st.slider("CPU Threads", 2, 24, 12)
ram = st.slider("RAM (GB)", 4, 64, 8)
storage = st.slider("Storage (GB)", 128, 2048, 256)
ssd = st.radio("Has SSD?", [0, 1])
hdd = st.radio("Has HDD?", [0, 1])
graphic_vram = st.slider("Graphic VRAM (GB)", 0, 12, 0)
battery = st.slider("Battery (WHr)", 30, 100, 70)
os = st.radio("Operating System", [0, 1])
screen_size = st.slider("Screen Size (inches)", 11.0, 18.0, 15.6)
ppi = st.slider("PPI", 100, 300, 200)
touchscreen = st.radio("Touchscreen", [0, 1])

# Predict Button
if st.button("Predict Price"):
    input_data = np.array([[brand, cpu_brand, cpu_gen, cpu_cores, cpu_threads,
                            ram, storage, ssd, hdd, graphic_vram, battery,
                            os, screen_size, ppi, touchscreen]])

    predicted_price = pipe.predict(input_data)[0]
    st.success(f"Predicted Laptop Price: ₹{int(predicted_price):,}")

    # Show nearby laptops
    st.subheader("💰 Similar Laptops:")
    low, high = predicted_price - 5000, predicted_price + 5000
    similar = laptop[laptop['Price'].between(low, high)]
    similar = similar[similar['Company'].str.lower() == brand.lower()]

    if not similar.empty:
        st.dataframe(similar[['Company', 'TypeName', 'Price', 'Cpu', 'Ram', 'Memory']])
    else:
        st.write("No similar laptops found.")
