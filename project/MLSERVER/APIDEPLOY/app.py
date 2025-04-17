import uvicorn
from fastapi import FastAPI
import joblib
from human_activity_record import HAR
import numpy as np
import json


app = FastAPI()
joblib_in = open("har-recommender.joblib","rb")
model=joblib.load(joblib_in)

@app.get('/')
def index():
    return {'message': 'Human Activity Recognition ML API'}


@app.post('/har/predict')
def predict_car_type(data:HAR):
    data = data.dict()
    print(data)
    json_str = json.dumps(data)
    json_object = json.loads(json_str)
    keys_list = list(json_object.keys())

    my_array = np.array([])

    for i in range(len(keys_list)):
       print(keys_list[i])
       value = json_object[keys_list[i]]
       print(value)
       my_array = np.append(my_array, value)

    print(my_array)

    arr = np.array(my_array, ndmin=2)
    print(arr.ndim) 
    print(arr.shape)

    print("starting to make prediction ...")
    prediction = model.predict(arr)
    print("Prediction completed")

    prediction_encoded = prediction[0]
    prediction_decoded = ''

    if prediction_encoded == 0:
        prediction_decoded = 'LAYING'
    elif prediction_encoded == 1:
        prediction_decoded = 'SITTING'
    elif prediction_encoded == 2:
        prediction_decoded = 'STANDING'
    elif prediction_encoded == 3 :
        prediction_decoded = 'WALKING'
    elif prediction_encoded == 4 :
        prediction_decoded = 'WALKING_DOWNSTAIRS'
    elif prediction_encoded == 5 :
        prediction_decoded = 'WALKING_UPSTAIRS'
    else :
        prediction_decoded = 'UNKNOWN'
    
    #Category Mapping: {'STANDING': 2, 'SITTING': 1, 'LAYING': 0, 'WALKING': 3, 'WALKING_DOWNSTAIRS': 4, 'WALKING_UPSTAIRS': 5}

    return {
        'prediction': prediction_decoded
    }



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)