from fastapi import FastAPI
import joblib
import numpy as np


model = joblib.load('xgb.joblib')

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'House Pricing ML API'}

@app.get('/predict')
def predict(data: dict):
    features = np.array(data['features'].reshape(1, -1))
    prediction = model.predict(features)
    return {'predicted_price': prediction}