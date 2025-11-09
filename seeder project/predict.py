import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.applications.resnet50 import ResNet50, preprocess_input
import joblib
import cv2
import os
from PIL import UnidentifiedImageError

# Load Models
cnn_model = keras.models.load_model('models/cnn_model.h5')
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
rf_model = joblib.load('models/rf_model.pkl')
mobilenet_model = keras.models.load_model('models/mobilenet_model.h5')

# Streamlit UI
def main():
    st.title("Paddy Straw Pest Detection")
    uploaded_file = st.file_uploader("Upload a Paddy Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        result = predict_image(uploaded_file)
        st.write("### Prediction Results:")
        for model_name, (prediction, confidence) in result.items():
            st.write(f"{model_name}: {prediction} (Confidence: {confidence * 100:.2f}%)")

# Image Prediction Function
def predict_image(uploaded_file):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        
        # CNN Prediction
        cnn_pred = cnn_model.predict(img)[0][0]
        cnn_result = 'Pest' if cnn_pred > 0.5 else 'No Pest'
        
        # ResNet Feature Extraction & RF Prediction
        features = resnet_model.predict(img)
        rf_pred = rf_model.predict(features.reshape(1, -1))[0]
        rf_result = 'Pest' if rf_pred == 1 else 'No Pest'
        
        # MobileNet Prediction
        mobilenet_pred = mobilenet_model.predict(img)[0][0]
        mobilenet_result = 'Pest' if mobilenet_pred > 0.5 else 'No Pest'
        
        return {
            "CNN Prediction": (cnn_result, cnn_pred if cnn_result == 'Pest' else 1 - cnn_pred),
            "Random Forest Prediction": (rf_result, 1.0),  # Assuming RF model gives binary output
            "MobileNet Prediction": (mobilenet_result, mobilenet_pred if mobilenet_result == 'Pest' else 1 - mobilenet_pred)
        }
    except UnidentifiedImageError:
        return "Error: The uploaded file is not a valid image. Please upload a valid image file."

if __name__ == "__main__":
    main()
