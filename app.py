%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="🌾 AI Crop Recommendation App")

@st.cache_data
def load_data():
    return pd.read_csv("crop_recommendation.csv")

df = load_data()

@st.cache_resource
def train_model(data):
    X = data.drop("label", axis=1)
    y = data["label"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model(df)

st.title("🌱 AI Crop Recommender")
st.markdown("Enter your soil & weather values:")

N = st.slider("Nitrogen", 0, 140, 70)
P = st.slider("Phosphorus", 0, 145, 60)
K = st.slider("Potassium", 0, 205, 70)
temperature = st.slider("Temperature (°C)", 10, 45, 25)
humidity = st.slider("Humidity (%)", 10, 100, 60)
ph = st.slider("pH", 3.0, 10.0, 6.5)
rainfall = st.slider("Rainfall (mm)", 20, 300, 100)

if st.button("🚀 Recommend Crop"):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(features)[0]
    st.success(f"✅ Recommended Crop: {prediction.capitalize()}")
