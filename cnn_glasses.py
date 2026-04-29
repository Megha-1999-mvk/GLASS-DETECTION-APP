from tensorflow.keras import backend as K
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
K.clear_session()

# Load trained model

def load_model():
    model = tf.keras.models.load_model("glasses.model.h5")  
    return model

model = load_model()

# App Title
st.title("👓 Glasses Detection App")
st.write("Upload an image to check whether the person is wearing glasses or not.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Class labels 
class_names = ["Glasses", "No Glasses"]

def preprocess_image(uploaded_file):
    file_bytes= np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    array_resize=cv2.resize(image,(100,100))
    array_reshape=array_resize.reshape(1,100,100,3)
    array_norm=array_reshape/255
    print(array_norm.shape)
    return array_norm

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    uploaded_file.seek(0)
    if st.button("Predict"):
        processed_img = preprocess_image(uploaded_file)
        prediction = model.predict(processed_img)

        prob = prediction[0]
        predicted_class = np.argmax(prob)
        confidence = np.max(prob)

        st.subheader("Prediction:")
        st.write(f"**{class_names[predicted_class]}**")

        st.subheader("Confidence Score:")
        st.write(f"{confidence * 100:.2f}%")