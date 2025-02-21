import os
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import pickle
from PIL import Image
import tempfile

# Function to load the pre-trained model
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Function to load the label encoder
def load_label_encoder(le_path):
    with open(le_path, 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    return label_encoder

# Function to preprocess the fingerprint image
def preprocess_fingerprint(image_path):
    # Open the image in grayscale using Pillow
    img = Image.open(image_path).convert("L")  # Open and convert to grayscale

    # Resize the image to (128, 128)
    img = img.resize((128, 128))

    # Convert to NumPy array and normalize pixel values
    img = np.array(img).astype('float32') / 255.0

    # Add channel dimension (grayscale image)
    img = np.expand_dims(img, axis=-1)  # Shape becomes (128, 128, 1)

    return img

# Function to predict the class of a fingerprint image
def predict_fingerprint(image_path, model, label_encoder):
    img = preprocess_fingerprint(image_path)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)

    # Decode the predicted class to the original label (blood group)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    return predicted_label[0]

# Streamlit app
def main():
    st.title("Fingerprint Image Classification with Pre-trained Model")

    # Load the model and label encoder once the app is run
    model_path = "fingerprint_model.h5"
    le_path = "label_encoder.pkl"

    if os.path.exists(model_path) and os.path.exists(le_path):
        model = load_model(model_path)
        label_encoder = load_label_encoder(le_path)
        st.success("Model and Label Encoder loaded successfully!")
    else:
        st.error("Model or Label Encoder not found! Please ensure they are in the correct directory.")
        return

    # Upload an image for prediction
    uploaded_file = st.file_uploader("Upload a fingerprint image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Save the uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            img_path = tmp_file.name

        # Display the uploaded image
        img = Image.open(img_path).convert("L")  # Open and convert to grayscale
        st.image(img, caption="Uploaded Fingerprint Image", use_container_width=True)

        # Prediction on the uploaded image
        if st.button("Predict"):
            predicted_label = predict_fingerprint(img_path, model, label_encoder)
            st.write(f"Predicted Blood Type: {predicted_label}")

if __name__ == "__main__":
    main()
