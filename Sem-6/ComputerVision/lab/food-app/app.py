import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os

# --- Load model and calorie map ---
model = load_model('kaggle_model/food_model.h5')
calorie_data = pd.read_csv('kaggle_model/calories.csv')
class_indices = sorted(calorie_data['food'].tolist())

# --- Helper Function ---
def predict_food(img):
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_batch = np.expand_dims(img_resized, axis=0)
    preds = model.predict(img_batch)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]
    predicted_class = class_indices[top_idx]
    return predicted_class, confidence

# --- Streamlit UI ---
st.set_page_config(page_title="Food Calorie Estimator", layout="centered")
st.title("🍔 Food Recognition & Calorie Estimation")
st.markdown("Upload a food image and get estimated calories!")

uploaded_file = st.file_uploader("Upload Food Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        food_name, conf = predict_food(img_array)
        calories = calorie_data[calorie_data['food'] == food_name]['calories'].values[0]

    st.success(f"🍽️ Predicted Food: **{food_name.replace('_', ' ').title()}**")
    st.info(f"🔥 Estimated Calories: **{calories} kcal**")
    st.progress(int(conf * 100))
