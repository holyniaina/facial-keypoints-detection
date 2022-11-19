import streamlit as st
import numpy as np
from PIL import Image
# import pickle
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.write("HELLO")

def load_image(image_file):
	img = Image.open(image_file)
	return img

def preprocess_image(image): 
  #resize image
  image_resized = cv2.resize(image, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
  #reshape image
  image_reshaped = image_resized.reshape(1,96,96,1)
  #scale image
  X = image_reshaped/255.
  return X

def predict_keypoints(img):
  model = load_model('my_streamlit_page/my_model')
  # model = pickle.load(open("my_streamlit_page/model_keypoints_detection.pkl","rb"))
  y_pred = model.predict(img)
  keypoints = y_pred[0].reshape(15, 2)
  return keypoints

def plot_keypoints(img, keypoints):
  fig, ax = plt.subplots(facecolor="#11101B")
  ax.imshow(img)
  ax.scatter(x=keypoints[:,0], y=keypoints[:,1], c='red')
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
  st.pyplot(fig)

st.title("Detect Keypoints")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
  # To View Uploaded Image
  st.image(load_image(image_file),width=250)

  # convert img to np_array
  img = Image .open(image_file)
  img_np = np.array(img)[:,:,0]
  y, x = img_np.shape

  processed_img = preprocess_image(img_np)
  keypoints = predict_keypoints(processed_img)
  # keypoints::: [[x,y],[x,y]]
  proportion = np.array([x/96, y/96]).reshape((1,-1))
  new_keypoints = keypoints * proportion
  # st.write(proportion)
  # st.write(keypoints)
  # st.write("\n\n")
  # st.write(keypoints * proportion)
  # st.write(keypoints)

  plot_keypoints(img, new_keypoints)

