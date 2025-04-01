import streamlit as st
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os

# --- Configuration ---
ARTIFACTS_DIR = "saved_artifacts"
IMAGES_DIR = "images"
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct absolute paths to artifacts relative to the app's location
ARTIFACTS_DIR_ABS = os.path.join(APP_DIR, "saved_artifacts") # Assumes saved_artifacts is in the same dir as app.py

# --- Load Artifacts (Cached for Performance) ---
@st.cache_resource
def load_sklearn_model(path):
    return joblib.load(path)

@st.cache_resource
def load_keras_model(path):
    return load_model(path, compile=False)

@st.cache_data
def load_joblib_data(path):
    return joblib.load(path)

# Construct full paths
# ... (paths remain the same) ...

ARTIFACTS_DIR=ARTIFACTS_DIR_ABS
log_reg_path = os.path.join(ARTIFACTS_DIR, "log_reg_model.joblib")
rf_model_path = os.path.join(ARTIFACTS_DIR, "rf_model.joblib")
lstm_model_path = os.path.join(ARTIFACTS_DIR, "lstm_model.h5")
label_encoder_path = os.path.join(ARTIFACTS_DIR, "label_encoder.joblib")
scaler_path = os.path.join(ARTIFACTS_DIR, "scaler.joblib")
final_columns_path = os.path.join(ARTIFACTS_DIR, "final_columns.joblib")
predefined_samples_path = os.path.join(ARTIFACTS_DIR, "predefined_samples.joblib")


# Load all artifacts
try:
    # ... (loading remains the same) ...
    log_reg_model = load_sklearn_model(log_reg_path)
    rf_model = load_sklearn_model(rf_model_path)
    lstm_model = load_keras_model(lstm_model_path)
    label_encoder = load_sklearn_model(label_encoder_path)
    scaler = load_sklearn_model(scaler_path)
    final_selected_columns = load_joblib_data(final_columns_path)
    predefined_samples = load_joblib_data(predefined_samples_path)
    N_SELECTED_FEATURES = len(final_selected_columns)

except FileNotFoundError as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during loading: {e}")
    st.stop()


# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")
model_options = ("Logistic Regression", "Random Forest", "LSTM")
selected_model_name = st.sidebar.selectbox("Choose a Model:", model_options)
st.sidebar.subheader("Select Predefined Sample")
sample_input_data_from_sidebar = None
selected_sample_key = ""
if not predefined_samples:
    st.sidebar.warning("No predefined samples loaded.")
else:
    sample_options = list(predefined_samples.keys())
    selected_sample_key = st.sidebar.selectbox("Choose a Sample:", sample_options)
    sample_input_data_from_sidebar = predefined_samples[selected_sample_key]
    st.sidebar.caption(f"Using sample for: {selected_sample_key.replace('Sample: ', '')}")


# --- Main Panel Display ---
st.title("🚶‍♂️ UCI HAR Activity Prediction 🚶‍♀️")
# --- Add Header Image ---
header_image_url = "https://images.pexels.com/photos/373543/pexels-photo-373543.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
st.markdown( """ <style> [data-testid="stImage"] > img { max-height: 250px; object-fit: cover; } </style> """, unsafe_allow_html=True )
try:
    st.image(header_image_url, use_container_width=True)
except Exception as img_e:
    st.warning(f"Could not load header image from URL. Error: {img_e}")
# --- End Header Image ---

st.write("Configure the model in the sidebar. Choose an input method below, then click Predict.")
st.write(f"(Models trained on {N_SELECTED_FEATURES} selected features)")
st.divider()

# --- Input Method Selection ---
st.subheader("Input Data")
input_method = st.radio(
    "Choose input method:",
    ("Use Predefined Sample (selected in sidebar)", "Upload CSV File"),
    key="input_method_radio"
)

# --- File Upload Logic ---
uploaded_file = None
input_df_features_only = None # Store features for prediction here
true_labels = None # Store true labels here if available
validation_error = None
upload_success = False # Flag for successful upload and validation

if input_method == "Upload CSV File":
    uploaded_file = st.file_uploader("Upload CSV (must include required features; 'Activity' column optional for comparison)", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df_from_file = pd.read_csv(uploaded_file)
            st.write("Uploaded DataFrame Preview:")
            st.dataframe(input_df_from_file.head())

            # --- Extract True Labels (if available) ---
            if 'Activity' in input_df_from_file.columns:
                true_labels = input_df_from_file['Activity'].copy() # Extract true labels
                # Prepare DataFrame with features only for validation/prediction
                input_df_features_only = input_df_from_file.drop('Activity', axis=1)
                st.info("Found 'Activity' column for comparison.")
            else:
                input_df_features_only = input_df_from_file.copy() # Use all columns as features
                st.warning("Optional 'Activity' column not found in uploaded file. Cannot show true labels.")

            # --- Validation (on feature columns only) ---
            missing_cols = set(final_selected_columns) - set(input_df_features_only.columns)
            if missing_cols:
                validation_error = f"Error: Uploaded CSV is missing required feature columns: {missing_cols}"
                st.error(validation_error)
                input_df_features_only = None # Invalidate features df
            else:
                # Select and reorder feature columns
                input_df_features_only = input_df_features_only[final_selected_columns]
                st.success(f"CSV structure validated ({input_df_features_only.shape[0]} rows). Ready for prediction.")
                upload_success = True # Mark as successful

        except Exception as e:
            validation_error = f"Error processing CSV file: {e}"
            st.error(validation_error)
            input_df_features_only = None

st.divider()

# --- Prediction Area ---
st.subheader("Run Prediction")

# Determine which data to use
data_to_predict = None
data_source_key = None
display_mode = "single"

if input_method == "Use Predefined Sample (selected in sidebar)":
    if sample_input_data_from_sidebar is not None:
        data_to_predict = sample_input_data_from_sidebar
        data_source_key = selected_sample_key
        display_mode = "single"
        true_labels = None # No true labels for predefined samples currently
    else:
        st.warning("Select a valid predefined sample from the sidebar.")

elif input_method == "Upload CSV File":
    if upload_success and input_df_features_only is not None:
        # Use the validated & ordered features DataFrame
        data_to_predict = input_df_features_only
        data_source_key = "Uploaded File"
        display_mode = "multi"
        # 'true_labels' variable was set during file processing
    elif uploaded_file is None:
        st.info("Upload a CSV file above to enable prediction.")
    # If uploaded_file is not None but validation failed, error was already shown

# Only show button if we have valid data to predict
if data_to_predict is not None and data_source_key is not None:
    if st.button(f"Predict Activity for {data_source_key}"):
        try:
            # --- Preprocessing ---
            if isinstance(data_to_predict, np.ndarray) and data_to_predict.ndim == 1:
                 data_to_predict_prepared = data_to_predict.reshape(1, -1)
            else:
                 data_to_predict_prepared = data_to_predict.values if isinstance(data_to_predict, pd.DataFrame) else data_to_predict

            scaled_data = scaler.transform(data_to_predict_prepared)
            num_samples = scaled_data.shape[0]

            # --- Prediction ---
            predictions_encoded = None
            with st.spinner(f"Predicting using {selected_model_name} for {num_samples} sample(s)..."):
                # ... (Prediction logic remains the same) ...
                if selected_model_name == "Logistic Regression":
                    predictions_encoded = log_reg_model.predict(scaled_data)
                elif selected_model_name == "Random Forest":
                    predictions_encoded = rf_model.predict(scaled_data)
                elif selected_model_name == "LSTM":
                    reshaped_data = scaled_data.reshape((num_samples, 1, N_SELECTED_FEATURES))
                    prediction_probs = lstm_model.predict(reshaped_data)
                    predictions_encoded = np.argmax(prediction_probs, axis=1)

            # --- Decode and Display Results ---
            if predictions_encoded is not None:
                predicted_activities = label_encoder.inverse_transform(predictions_encoded)
                st.subheader("Prediction Results")

                if display_mode == "single":
                    # Display single result with image (no true label comparison here)
                    activity_name = predicted_activities[0]
                    st.success(f"The predicted activity is: **{activity_name}**")
                    image_filename = f"{activity_name}.jpeg"
                    image_path = os.path.join(IMAGES_DIR, image_filename)
                    col1, col2 = st.columns([0.7, 0.3])
                    with col2:
                        if os.path.exists(image_path):
                            st.image(image_path, caption=f"{activity_name}", width=150)
                        else:
                            st.warning(f"Img not found: {image_filename}")
                else: # display_mode == "multi"
                    # Create results DataFrame with predicted activity
                    results_df = pd.DataFrame({'Predicted_Activity': predicted_activities})

                    # Add True Activity if available from upload
                    if true_labels is not None:
                         results_df['True_Activity'] = true_labels.reset_index(drop=True)
                         # Reorder columns for clarity
                         results_df = results_df[['True_Activity', 'Predicted_Activity']]

                    # Combine with first few input features for context
                    # Need to use the DataFrame before .values conversion if possible, or recreate
                    input_features_df = data_to_predict if isinstance(data_to_predict, pd.DataFrame) else pd.DataFrame(data_to_predict_prepared, columns=final_selected_columns)
                    display_df = pd.concat([input_features_df.reset_index(drop=True).iloc[:, :5], results_df], axis=1)

                    st.dataframe(display_df)
                    # Provide download option for results (now includes True_Activity if present)
                    csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button( label="Download Results as CSV", data=csv, file_name='har_predictions.csv', mime='text/csv')
            else:
                st.error("Could not make predictions.")
        except Exception as pred_e:
            st.error(f"An error occurred during prediction: {pred_e}")


# Footer Info
st.divider()
st.info(f"App requires a CSV with {N_SELECTED_FEATURES} features (column order doesn't matter if names match). Optional 'Activity' column can be included for comparison.")