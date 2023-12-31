# Required Libraries
import tensorflow as tf
import streamlit as st
import keras
from keras.models import load_model
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from tensorflow.python import pywrap_tensorflow as _pywrap_tensorflow

# Load Model
model = load_model('mnist_cnn_model.h5',compile = False)

# Function to Transform Input
def transform_image(img):
    # Convert image to grayscale and resize it
    img = img.convert('L').resize((28, 28))
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img.astype('float32')
    img /= 255
    return img

# Streamlit App
st.title("MNIST Digit Classification using CNN")
st.write("Draw a digit below and click 'Predict'")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 1)",  # Drawing color
    stroke_width=20,  # Stroke width
    stroke_color="rgba(255, 165, 0, 1)",
    background_color="rgba(0, 0, 0, 1)",  # Canvas background
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button('Predict'):
    if canvas_result.image_data is not None:
        # Convert the image data to PIL Image
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        st.image(img, use_column_width=False, caption='Input Image')
        img = transform_image(img)
        prediction = np.argmax(model.predict(img), axis=-1)
       # st.write("Predicted digit: ", prediction[0])
        st.markdown(f"<h2 style='color: white'>Predicted digit: {prediction[0]}</h2>", unsafe_allow_html=True)
    else:
        st.write("Draw a digit in the canvas first.")
