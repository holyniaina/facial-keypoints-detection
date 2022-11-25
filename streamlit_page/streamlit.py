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

def plot_keypoints(img, keypoints, grid):
  fig, ax = plt.subplots(facecolor="#11101B")
  ax.imshow(img)
  ax.scatter(x=keypoints[:,0], y=keypoints[:,1], c='red')
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_visible(False)
  grid.pyplot(fig)
  grid.write('Keypoints')
  

def start_process(image, grid):
  image_original_grey = image.convert('L')
  processed_img = preprocess_image(image_original_grey)
  keypoints = predict_keypoints(processed_img)
  new_keypoints = get_new_keypoints(image_original_grey, keypoints)
  plot_keypoints(image, new_keypoints, grid)

CSS = """
.block-container { max-width: 60vw; padding: 0.5rem 0 5rem 0; }
div.row-widget.stButton { font-size: 1.5rem; margin-top: 50%; }
[data-testid="stImage"] { width: 100%; }
[data-testid="stMarkdownContainer"] > p { width: 100%; text-align: center; font-size: 1.3rem; }
"""

st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)

st.markdown("""# Facial Keypoints Detection
## Batch 919 - Remote
---""")
image_file = st.file_uploader("Upload Your Image", type=["png","jpg","jpeg"])
grid = st.columns((2, 1, 2))
if image_file:
  button_pressed = grid[1].button('Detect keypoint')
  image = Image.open(image_file)
  grid[0].image(image)
  grid[0].write('Default Image')
  if button_pressed:
    CSS2 = """div.row-widget.stButton{ display: none !important; }"""
    start_process(image, grid[2])
    st.write(f'<style>{CSS2}</style>', unsafe_allow_html=True)
    st.markdown("""---""")
