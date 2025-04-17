This file contains steps and scripts details to be run to reproduce our work for <b>
Human Activity Recognition for Health Monitoring Using Wearable Devices by group 7</b> <br>

Git Hub Repo(Public) : https://github.com/preethibosco/HumanActivityRecognition<br>
Link to dataset used https://github.com/preethibosco/HumanActivityRecognition/tree/main/DataSets  <br>

<b>Instrcution to load & train model</b>

<b>Part 1(UCI HAR) :</b><br>
Data engineering, feature engineering , Model training, Model evaluation with UCI dataset <br>
https://github.com/preethibosco/HumanActivityRecognition/tree/main/DataSets/UCI-HAR  <br>
Run the notebook @ <br>
https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/model-training/HAR-LOGREG-LSTM.ipynb <br>

<b>Part 2(Capture24) :</b><br>
Data engineering, feature engineering , Model training, Model evaluation with capture24 dataset <br>
https://github.com/preethibosco/HumanActivityRecognition/tree/main/DataSets/Capture24 <br>
Run the notebook @ <br>
https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/model-training/Capture24.ipynb <br>

<b>Part 3(WISDM) :</b><br>
Data engineering, feature engineering , Model training, Model evaluation with WISDM dataset <br>
https://github.com/preethibosco/HumanActivityRecognition/tree/main/DataSets/WISDM <br>
Run the notebook @ <br>
https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/model-training/WISDM-RF-XG-LR.ipynb <br>


<b>Deployment</b>

<b>Part 4(Application - stremlit) :</b><br>
Run the model training & serialise and store learned model as joblib file.<br>
https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/MLSERVER/UCI_Ver2.ipynb <br>

Clone the folder from <br>
https://github.com/preethibosco/HumanActivityRecognition/tree/main/project/MLSERVER <br>

Create python virtual environment with requirement file @ https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/MLSERVER/requirements.txt <br>

Store data set in current folder <br>
Run https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/MLSERVER/csv_generation.py to generate randomised sample for testing <br>

Run & start streamlit app with following command:

https://github.com/preethibosco/HumanActivityRecognition/blob/main/project/MLSERVER/app.py
streamlit run app.py


<b>Part 5(API -FastAPI):</b><br>

To perform online inferencing, we need following steps

| Step # | Step | 
| ---    | ---- | 
| 1 |  Save the model in a serialization format, we used Joblib |
| 2 |  Develop a FastAPI application with an API route, and in the logic, load the model, take JSON request which is validated by Python Pydantic and use the serialized model to inference and reply |
| 3 |  Run the FastAPI app and using any REST client tool (example Insomnia) - try out a request and response |

#### 1. Save the model in a serialization format, we used Joblib
---

