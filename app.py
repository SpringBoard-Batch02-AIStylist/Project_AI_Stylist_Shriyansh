import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D
from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        color: #4CAF50;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5em;
        color: #4CAF50;
        text-align: center;
    }
    .image-container {
        display: flex;
        justify-content: center;
        padding: 10px;
    }
    .image-caption {
        text-align: center;
        font-size: 1em;
        color: #555;
    }
    </style>
    """, unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">Personal AI Stylist</h1>', unsafe_allow_html=True)

# Load pre-trained models and data
Image_features = pkl.load(open('Features_of_Images.pkl', 'rb'))
Filenames = pkl.load(open('filenames.pkl', 'rb'))

def image_feature_extraction(image_path, model):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.models.Sequential([model, GlobalMaxPool2D()])

# KNN for image retrieval
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)

# File uploader
upload_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # Save uploaded file
    with open(os.path.join('upload', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())

    # Display uploaded image
    st.markdown('<h2 class="sub-header">Uploaded Image</h2>', unsafe_allow_html=True)
    st.image(upload_file, use_column_width=True, caption="Your Uploaded Image")

    # Extract features and find nearest neighbors
    with st.spinner('Processing...'):
        input_img_features = image_feature_extraction(upload_file, model)
        distance, indices = neighbors.kneighbors([input_img_features])
        st.success('Processing Complete!')

    # Display recommended images
    st.markdown('<h2 class="sub-header">Recommended Images for You</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(Filenames[indices[0][1]], caption="Recommendation 1", use_column_width=True)

    with col2:
        st.image(Filenames[indices[0][2]], caption="Recommendation 2", use_column_width=True)

    with col3:
        st.image(Filenames[indices[0][3]], caption="Recommendation 3", use_column_width=True)

    with col4:
        st.image(Filenames[indices[0][4]], caption="Recommendation 4", use_column_width=True)

    with col5:
        st.image(Filenames[indices[0][5]], caption="Recommendation 5", use_column_width=True)
