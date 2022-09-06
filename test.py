import pickle
import numpy as np
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2


feature_list = np.array(pickle.load(open('features.pkl','rb')))
#print((feature_list).shape)
filenames = pickle.load(open('filenames.pkl','rb'))

#Repeating the steps from app.py:
#create the model:
model = ResNet50(weights="imagenet",include_top=False,input_shape=(224,224,3))
model.trainable = False

#adding our output layer
model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

#testing the output:
img = image.load_img('sample/kitchen_819.jpg', target_size=(224, 224))
# convert the image to array
img_array = image.img_to_array(img)
# expand the dims to give batch of arrays
expanded_img_array = np.expand_dims(img_array, axis=0)
# preprocessing the image input for RestNet model
preprocessed_img = preprocess_input(expanded_img_array)
# predicting and converting to 1D
result = model.predict(preprocessed_img).flatten()
# normalising the result to get values between 0 to 1
normalized_result = result / norm(result)


#use Nearest Neighbors to get similar images as the output:
neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

#extracting distances and indices:
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

#get the image name from indices
for file in indices[0]:
    #print(filenames[file])
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output',temp_img)
    cv2.waitKey(0)

