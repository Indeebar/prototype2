import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Title
st.title("🌾 AgriWaste Quality Classifier ")

# Load model once
@st.cache_resource
def load_classifier():
    return load_model("agriwaste_classifier_model.h5")

model = load_classifier()

# Define class labels
class_labels = [
    'Banana_Stems_Contaminated', 'Banana_Stems_Dry', 'Banana_Stems_Moisturized',
    'Mustard_Stalk_Moist', 'Apple_Pomace_Dry', 'Cashewnut_Shells_Dry',
    'Coconut_shells_Dry', 'Groundnut_Shells_Dry', 'Jute_Stalks_Dry',
    'Maize_Stalks_Dry', 'Pineapple_Leaves_Dry', 'Rice_Straw_Dry',
    'Sugarcane_bagasse_Dry', 'Wheat_Straw_Dry'
]

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"✅ Predicted Class: **{pred_class}**")
    st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")
