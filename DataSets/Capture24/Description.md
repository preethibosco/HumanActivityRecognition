Capture 24 : 


The Capture 24 dataset is a comprehensive collection of wrist-worn accelerometer data, designed to enhance research in human activity recognition, mood analysis, and health monitoring. Collected between 2014 and 2016, it comprises data from 151 participants in the Oxfordshire area, each wearing an Axivity AX3 device for approximately 24 hours, totaling nearly 4,000 hours of data. To obtain ground truth annotations, participants also wore Vicon Autograph wearable cameras during daytime and used Whitehall II sleep diaries to register their sleep times, resulting in more than 2,500 hours of labeled data.

Contents of the Dataset:

Accelerometer Data: Raw 3-axis accelerometer readings (X, Y, Z) with timestamps.
Annotated Activities: Labels indicating activities such as walking, sitting, running, and sleeping.
Sleep Diaries: Information on sleep patterns, durations, and quality.
Wearable Camera Data: Images captured to provide contextual ground truth for activities.

File Structure:
The dataset is organized into directories for each participant, containing:

Accelerometer Files: Typically in .csv.gz formats, storing raw accelerometer data.
Annotation Files: Files detailing activity labels and timestamps.
Sleep Logs: Documents or spreadsheets with sleep-related information.
Image Data: Folders containing images from wearable cameras.

Data Sources & Files Used:
Metadata: Loaded from metadata.csv, containing general dataset information.
Activity Annotations: Loaded from annotation-label-dictionary.csv, mapping activity labels to their descriptions.
Sensor Data Files: Extracted from multiple .csv.gz files in the /Users/vaijanatha/Downloads/capture24 directory.

Data Preprocessing Steps:
Checked for missing values before processing.
Ensured necessary columns exist (timestamp, x, y, z).
Converted timestamp to datetime format.
Forward-filled missing values.
Verified missing values after handling.
Feature Normalization:
Used StandardScaler to normalize accelerometer readings (x, y, z).
Data Verification:
Displayed missing value statistics after preprocessing.
Printed the first few rows of the processed dataset.

Data Visualization:
Accelerometer Readings Over Time:
Plotted x, y, and z acceleration values over time.
Used line plots to analyze acceleration trends in different axes.
Added grid and labels for clarity.
Activity Distribution:

Counted occurrences of simplified activities in the dataset.
Created a bar plot showing the top 15 most frequent activities using Seaborn.
Used a "viridis" color palette for better visual differentiation.

Machine Learning - Logistic Regression for Activity Classification:
Feature Selection:
Used x, y, and z accelerometer readings as input features (X).
The extracted activity column served as the target variable (y).
Data Preprocessing for Model Training:

Converted categorical activity labels into numerical codes using astype('category').cat.codes.
Split the dataset into training (80%) and testing (20%) sets using train_test_split.
Model Training & Evaluation:

Trained a Logistic Regression model on the training set.
Predicted activities on the test set.
Evaluated the model using accuracy score, achieving 45% accuracy.

Conclusion:
The data was successfully cleaned, simplified, and visualized.
A logistic regression model was trained for activity classification based on accelerometer data.
The achieved accuracy (45%) suggests the need for further feature engineering or advanced models (e.g., Decision Trees, Random Forest, Deep Learning).
After feature engg and model comparision xgboost gives the highest accuracy of 80%
