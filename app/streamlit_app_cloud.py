import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.title("Classification des maladies des feuilles")

# Charger le modèle
model = load_model("../model/model.h5")

# Classes (doivent correspondre à l'entraînement)
classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
           'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 
           'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
           'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 
           'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 
           'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 
           'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
           'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 
           'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 
           'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 
           'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 
           'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
           'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
           'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']

uploaded_file = st.file_uploader(
    "Uploader une image de feuille",
    type=["jpg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, width=300)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    st.success(f"🌱 Diagnostic : **{predicted_class}**")
