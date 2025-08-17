import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load Model
model = tf.keras.models.load_model(r"D:\CNN-Potato-Disease\My-App\model.h5")
class_names = ['Early Blight', 'Late Blight', 'Healthy']

# Page Config
st.set_page_config(page_title="Potato Disease Detector", page_icon="🥔", layout="centered")

# Background Image CSS
background_url = "https://images.pexels.com/photos/4750270/pexels-photo-4750270.jpeg"
st.markdown(f"""
    <style>
    .stApp {{
        background: url("{background_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(255,255,255,0.7);
        padding: 2rem;
        border-radius: 15px;
    }}
    h1 {{
        font-size: 3em;
        color: #1b5e20;
        text-align: center;
        font-weight: bold;
    }}
    .choose-text {{
        font-size: 1.5em;
        text-align: center;
        margin-bottom: 10px;
        color: #2e7d32;
        font-weight: bold;
    }}
    .result-card {{
        background-color: rgba(255,255,255,0.85);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
    }}
    .footer {{
        text-align: center;
        color: #1b5e20;
        font-weight: bold;
        margin-top: 40px;
    }}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🥔 Potato Disease Detector</h1>", unsafe_allow_html=True)
st.write("---")

# Upload Section
st.markdown('<div class="choose-text">📷 Choose an Image to Predict:</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption='Uploaded Image', width=300)

    img = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    with st.spinner('🔍 Analyzing the image...'):
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = round(100 * (np.max(prediction[0])), 2)

    st.write("---")
    st.markdown(f"<h3 style='text-align: center; color: #1b5e20;'>Prediction Results</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"<div class='result-card'>🌱 <b>Class:</b> {predicted_class}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='result-card'>📊 <b>Confidence:</b> {confidence}%</div>", unsafe_allow_html=True)
    st.progress(confidence / 100)

else:
    st.info("⬆️ Upload an image above to start prediction.")

st.write("---")
st.markdown("<div class='footer'>Developed by Eyad 🚀</div>", unsafe_allow_html=True)
