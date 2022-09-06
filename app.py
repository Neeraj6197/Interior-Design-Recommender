import tensorflow as tf
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
import os
from tqdm import tqdm
import pickle

#create the model:
model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable = False

#adding our output layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# print(model.summary())

#defining a function to extract the features from the image:

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

#storing the filenames:
filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
# print(len(filenames))
# print(filenames[0:5])

#storing the features
features_list = []
for file in tqdm(filenames):
    features_list.append(extract_features(file,model))

#print(np.array(features_list).shape)

#exporting the files:
pickle.dump(features_list,open('features.pkl','wb'))
pickle.dump(filenames,open('filenames.pkl','wb'))