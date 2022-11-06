import streamlit as st
import pandas as pd
import numpy as np
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
import joblib
import cv2


knn = joblib.load("D:\HT\hk7\ML\KNN\knn_digit.pkl")

st.title('Uber pickups in NYC')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    image = Image.open(uploaded_file)
    image = np.array(image)
    #image = Image.frombuffer("L",(28,28),bytes_data,'raw',"L", 0, 1)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image,(28,28))
    image = image.reshape(1,784)
    st.image(uploaded_file)
    #print(bytes_data)
    predicted = knn.predict(image)
    text = 'Kết quả: ' + str(predicted[0])
    st.subheader(text)

print("DDD")