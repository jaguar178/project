import tensorflow as tf
import numpy as np
import os

# absoluter Pfad (wichtig!)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "keras_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model NOT found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict_image(image):
    # Beispiel-Preprocessing (Teachable Machine kompatibel)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    
    confidence = float(np.max(preds))
    label_index = int(np.argmax(preds))

    return label_index, confidence
