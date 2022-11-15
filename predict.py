import requests

url = 'http://localhost:8000/predict'

params = {
    'file' :'raw_data/face.png'
}

# def predict_keypoints(X):
#   model = pickle.load(open("model_keypoints_detection.pkl","rb"))
#   y_pred = model.predict(X)
#   return y_pred

response = requests.post(url, params=params)
print(response.json())
