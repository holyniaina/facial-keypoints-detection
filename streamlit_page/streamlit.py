import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


def load_image(image_file):
    img = Image.open(image_file)
    return img

def preprocess_image(image):
  im_resized = image.resize((96,96))
  img_np = np.array(im_resized)
  image_reshaped = img_np.reshape(1,96,96,1)
  X = image_reshaped/255.
  return X

def get_coordinates(df_target):
  coords = []
  for i in range(len(df_target)):
    for j in range(0,df_target.shape[1],2):
      coords.append([df_target.iloc[i,j],df_target.iloc[i,j+1]])
  coords = np.array(coords)
  return coords

def predict_keypoints(img):
  model = load_model("model/model_beta")
  y_pred = model.predict(img, verbose=0)
  coords = get_coordinates(pd.DataFrame(y_pred))
  return coords

def get_new_keypoints(img, old_keypoints):
  # modify keypoints to correspond to original image
  y, x = np.array(img).shape
  proportion = np.array([x/96, y/96]).reshape((1,-1))
  new_keypoints = old_keypoints * proportion
  return new_keypoints

def plot_keypoints(img, keypoints):
  fig, ax = plt.subplots(facecolor="#11101B")
  ax.imshow(img)
  ax.scatter(x=keypoints[:,0], y=keypoints[:,1], c='red')
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
  st.pyplot(fig)


st.markdown("""# Batch 919 - Remote
## Facial Keypoints Detection
---""")
image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:
  st.image(load_image(image_file),width=250)

  image_original = Image.open(image_file)
  image_original_grey = image_original.convert('L')
  
  processed_img = preprocess_image(image_original_grey)
  keypoints = predict_keypoints(processed_img)
  new_keypoints = get_new_keypoints(image_original_grey, keypoints)
  plot_keypoints(image_original, new_keypoints)

