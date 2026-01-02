import streamlit as st
import requests

st.title("Classification des maladies des feuilles")

uploaded_file = st.file_uploader("Uploader une image de feuille", type=["jpg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Image uploadée", use_column_width=True)
    
    # Appel à l'API
    response = requests.post("http://localhost:8000/predict", files={"file": uploaded_file})
    st.write("Prédiction :", response.json()["prediction"])
