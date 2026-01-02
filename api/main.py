from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Chargeons le modèle sauvegardé
model = load_model("../model/model.h5")

# Liste des classes
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    class_idx = int(np.argmax(preds))

    return {"prediction": classes[class_idx]}

# Point d'entrée
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get["PORT", 8000])
    uvicorn.run(app, host="0.0.0.0", port=port)