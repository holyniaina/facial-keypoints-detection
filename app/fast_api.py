import cv2

from fastapi import FastAPI, Request
import base64
import numpy as np
import pickle


app = FastAPI()

def predict_keypoints(img):
    model = pickle.load(open("./model_keypoints_detection.pkl","rb"))
    y_pred = model.predict(img)
    keypoints = y_pred[0].reshape(15, 2)
    return keypoints


def preprocess_image(image): 
    #resize image
    image_resized = cv2.resize(image, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
    #reshape image
    image_reshaped = image_resized.reshape(1,96,96,1)
    #scale image
    X = image_reshaped/255.
    return X

@app.post("/predict_keypoints") 
<<<<<<< HEAD
def prediction_start(file: str):
    return {"file_size": len(file)}
=======
async def prediction_start(request: Request):
    return await request.json()

>>>>>>> 7a86ba5a8eedb32a695c7d05ad4ffce4cf86c38b

@app.get("/")
def read_root():
    return {"Hello": "World"}

# contents = await file.read()
# nparr = np.fromstring(contents, np.uint8)
# img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# image = file.file
# image_preprocess = preprocess_image(image)
# image_predict_keypoints = predict_keypoints(image_preprocess)
# _, encoded_img = cv2.imencode('.PNG', image_preprocess)
# encoded_img = base64.b64encode(encoded_img)