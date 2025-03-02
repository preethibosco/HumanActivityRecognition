## Solution Architecture
---


Common classification algorithms used for Human Activity Recognition (HAR) include: 

- `Support Vector Machines (SVM)` 
- `K-Nearest Neighbors (KNN)`
- `Random Forest Decision Trees` 
- `Logistic Regression`
- `Naive Bayes`
- `Convolutional Neural Networks (CNNs)`, with CNNs often considered the most effective for complex activity recognition due to their ability to extract features from time-series sensor data effectively
- `LSTM`

The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (`WALKING`, `WALKINGUPSTAIRS`, `WALKINGDOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded `accelerometer` and `gyroscope`, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. The experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data

In the solution architecture, we would like to have following, 

- Name of our product is `Sapphire Wellness`
- The product architecture
  - An User Interface where each user can login to view
     - Historical data
     - Classification data
  - A Data Application to serve the User Interface
  - Data Collection and Preparation
     - Data Storage and Ingestion
     - Data Exploration and Preparation
     - Data Labeling (if required)
     - Feature Store
  - Model Development & Training
  - Deployment (online ML Services)
  - Continuous Model and Monitoring
