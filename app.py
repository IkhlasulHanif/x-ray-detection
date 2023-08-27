import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model('./model.hdf5')

st.title('Fire Detection Image')

uploaded_file = st.file_uploader("Choose an image: ", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((256, 256))  
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    if st.button('Predict'):
        prediction = model.predict(image_batch)
        predicted_class_index = np.argmax(prediction)
        class_labels = {0: 'COVID19', 1: 'NORMAL', 2: 'PNEUMONIA', 3: 'TURBERCULOSIS'}
        predicted_class_label = class_labels[predicted_class_index]
        st.write(predicted_class_label)
