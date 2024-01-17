import streamlit as st
import numpy as np
import tensorflow as tf

# Check TensorFlow version
tf_version = tf.__version__
st.write(f"Using TensorFlow version: {tf_version}")

# Load models without compiling
fake_model = tf.keras.models.load_model('finalmodel.h5', compile=False)
real_model = tf.keras.models.load_model('my_model.h5', compile=False)

# Recompile with a compatible optimizer
fake_model.compile(optimizer='adam')  # Replace with a compatible optimizer if needed
real_model.compile(optimizer='adam')  # Replace with a compatible optimizer if needed

# Create a wrapper model to handle different input shapes
class WrapperModel:
    def __init__(self, real_model, fake_model):
        self.real_model = real_model
        self.fake_model = fake_model

    def predict(self, image):
        resized_image = image.resize((224, 224)) if self.real_model is not None else image.resize((256, 256))
        array = tf.keras.preprocessing.image.img_to_array(resized_image)
        array = np.expand_dims(array, axis=0)
        if self.real_model is not None:
            return self.real_model.predict(array)
        else:
            return self.fake_model.predict(array)

# Create the wrapper model
wrapper_model = WrapperModel(real_model, fake_model)

st.title("Fake Image Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = tf.keras.preprocessing.image.load_img(uploaded_file)  # Load image without resizing
    prediction = wrapper_model.predict(image)  # Let the wrapper handle resizing and prediction

    # Combine predictions (adjust logic if needed)
    final_prediction = "Fake" if prediction > 0.5 else "Real"  # Example threshold for single output value

    st.image(uploaded_file, caption=f"Prediction: {final_prediction}")
