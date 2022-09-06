import streamlit as st
import os
from PIL import Image
import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors

#importing the filename and featurelist:
feature_list = np.array(pickle.load(open('features.pkl','rb')))
#print((feature_list).shape)
filenames = pickle.load(open('filenames.pkl','rb'))

#create the model:
model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable = False

#adding our output layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#creating the title:
st.title('Interior Design Recommendations')

#creating a func to save the uploaded file:
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
            return 1
    except:
        return 0

def extract_features(img_path,model):
    #load the image
    img = image.load_img(img_path,target_size=(224,224))
    #convert the image to array
    img_array = image.img_to_array(img)
    #expand the dims to give batch of arrays
    expanded_img_array = np.expand_dims(img_array,axis=0)
    #preprocessing the image input for RestNet model
    preprocessed_img = preprocess_input(expanded_img_array)
    #predicting and converting to 1D
    result = model.predict(preprocessed_img).flatten()
    #normalising the result to get values between 0 to 1
    normalized_result = result/norm(result)
    #return the result
    return normalized_result

#creating a func to get the recommendations:
def recommend(features,feature_list):
    # use Nearest Neighbors to get similar images as the output:
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    # extracting distances and indices:
    distances, indices = neighbors.kneighbors([features])
    return indices

#uploading the file:
uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        #display the file
        display_image = Image.open(uploaded_file)
        st.image(display_image)

        #feature extraction:
        features = extract_features(os.path.join("uploads",uploaded_file.name),model)

        #recommendations:
        indices = recommend(features,feature_list)

        #displaying the images:
        st.text('Results related to your uploaded image:')
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])
        with col2:
            st.image(filenames[indices[0][1]])
        with col3:
            st.image(filenames[indices[0][2]])
        with col4:
            st.image(filenames[indices[0][3]])
        with col5:
            st.image(filenames[indices[0][4]])

    else:
        "Please upload a valid image file."