import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("models/cat_dog_model.h5")

st.title("Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img = np.array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        st.write("Prediction: Dog 🐶")
        st.write("Confidence:", prediction)
    else:
        st.write("Prediction: Cat 🐱")
        st.write("Confidence:", 1-prediction)
