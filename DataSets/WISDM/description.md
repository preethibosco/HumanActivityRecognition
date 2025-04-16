WISDM:
The WISDM (Wireless Sensor Data Mining) dataset is a time-series collection of accelerometer data captured via smartphones placed in participants' front pockets. It records 3-axis accelerometer readings while performing activities like walking, jogging, sitting, standing, upstairs, and downstairs, sampled at 20 Hz.

Dataset Contents:
Accelerometer Data: 3-axis (X, Y, Z) readings with timestamps, user IDs, and activity labels.

Activity Labels: Walking, Jogging, Sitting, Standing, Upstairs, Downstairs.

Preprocessing:
Removed missing and malformed values.

Used a sliding window (50 samples) to extract statistical features (mean, std, min, max, skewness, kurtosis, resultant acceleration).

Applied mean imputation and standardized features.

Machine Learning:
Features: Extracted statistical values from windows.

Models: Logistic Regression, Random Forest, XGBoost.

Train-Test Split: 80/20 stratified.

Results:
Random Forest achieved the highest accuracy of 96.1%, outperforming Logistic Regression and XGBoost.

Conclusion:
Random Forest proved highly effective for activity recognition on the WISDM dataset after cleaning, feature engineering, and model comparison.
