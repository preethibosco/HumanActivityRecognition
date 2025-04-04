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
# Assumes saved_artifacts and images are in the same directory as app.py
# If they are elsewhere, adjust the paths accordingly.
# Example: If they are one level up: APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR_ABS = os.path.join(APP_DIR, ARTIFACTS_DIR)
IMAGES_DIR_ABS = os.path.join(APP_DIR, IMAGES_DIR) # Define absolute path for images too

# --- Load Artifacts (Cached for Performance) ---
@st.cache_resource
def load_sklearn_model(path):
    """Loads a scikit-learn model using joblib."""
    return joblib.load(path)

@st.cache_resource
def load_keras_model(path):
    """Loads a Keras model."""
    # The compile=False argument is often useful when loading models saved without
    # their optimizer state, especially for inference-only tasks.
    return load_model(path, compile=False)

@st.cache_data
def load_joblib_data(path):
    """Loads general data saved with joblib."""
    return joblib.load(path)

# Construct full paths to artifact files
log_reg_path = os.path.join(ARTIFACTS_DIR_ABS, "log_reg_model.joblib")
rf_model_path = os.path.join(ARTIFACTS_DIR_ABS, "rf_model.joblib")
lstm_model_path = os.path.join(ARTIFACTS_DIR_ABS, "lstm_model.h5")
label_encoder_path = os.path.join(ARTIFACTS_DIR_ABS, "label_encoder.joblib")
scaler_path = os.path.join(ARTIFACTS_DIR_ABS, "scaler.joblib")
final_columns_path = os.path.join(ARTIFACTS_DIR_ABS, "final_columns.joblib")
predefined_samples_path = os.path.join(ARTIFACTS_DIR_ABS, "predefined_samples.joblib")

# Load all artifacts with error handling
try:
    log_reg_model = load_sklearn_model(log_reg_path)
    rf_model = load_sklearn_model(rf_model_path)
    lstm_model = load_keras_model(lstm_model_path)
    label_encoder = load_sklearn_model(label_encoder_path)
    scaler = load_sklearn_model(scaler_path)
    final_selected_columns = load_joblib_data(final_columns_path)
    predefined_samples = load_joblib_data(predefined_samples_path)
    N_SELECTED_FEATURES = len(final_selected_columns)

    # Check if predefined samples loaded correctly
    if not isinstance(predefined_samples, dict) or not predefined_samples:
        st.warning("Predefined samples did not load correctly or are empty.")
        predefined_samples = {} # Ensure it's a dict to avoid errors later

except FileNotFoundError as e:
    st.error(f"Error loading artifacts: Required file not found. {e}")
    st.error(f"Please ensure the '{ARTIFACTS_DIR}' directory exists in the same directory as the app script and contains all necessary .joblib and .h5 files.")
    st.stop() # Stop execution if essential files are missing
except Exception as e:
    st.error(f"An unexpected error occurred during artifact loading: {e}")
    st.stop()

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Configuration")

# 1. Model Selection
model_options = ("Logistic Regression", "Random Forest", "LSTM")
selected_model_name = st.sidebar.selectbox(
    "1. Choose a Model:",
    model_options,
    key="model_select"
)

# 2. Input Method Selection
input_method = st.sidebar.radio(
    "2. Choose Input Method:",
    ("Use Predefined Sample", "Upload CSV File"),
    key="input_method_radio"
)

st.sidebar.info(f"Using **{selected_model_name}** model.")
st.sidebar.info(f"Input method: **{input_method}**")


# --- Main Panel Display ---
st.title("🚶‍♂️ UCI HAR Activity Prediction 🚶‍♀️")

# --- Add Header Image ---
# Using a placeholder URL, replace with your actual image URL if needed
header_image_url = "https://images.theconversation.com/files/505850/original/file-20230123-14-iwjhhc.jpg?ixlib=rb-4.1.0&q=45&auto=format&w=926&fit=clip"
# Custom CSS to control image height and fit
st.markdown(
    """
    <style>
    [data-testid="stImage"] > img {
        max-height: 250px; /* Adjust height as needed */
        object-fit: cover; /* Or 'contain', depending on desired scaling */
    }
    </style>
    """,
    unsafe_allow_html=True
)
try:
    # Attempt to display the image, centered using container width
    st.image(header_image_url, use_container_width=True, caption="Human Activity Recognition")
except Exception as img_e:
    st.warning(f"Could not load header image from URL. Error: {img_e}")
# --- End Header Image ---

st.write(f"Configure the model and input method in the sidebar. Models trained on **{N_SELECTED_FEATURES}** selected features.")
st.divider()

# --- Input Area (Conditional based on Sidebar Selection) ---
st.subheader("Input Data")

data_to_predict = None
data_source_key = None
display_mode = "single" # Default, will change for CSV
true_labels = None # Store true labels here if available from CSV
upload_success = False # Flag for successful upload and validation
validation_error = None
selected_sample_key = None # Initialize

# --- Predefined Sample Selection (Main Page) ---
if input_method == "Use Predefined Sample":
    if not predefined_samples:
        st.warning("No predefined samples were loaded. Cannot use this option.")
    else:
        sample_options = ["--- Select Sample ---"] + list(predefined_samples.keys())
        selected_sample_key = st.selectbox(
            "Choose a Predefined Sample:",
            sample_options,
            key="sample_selectbox"
        )
        if selected_sample_key != "--- Select Sample ---":
            sample_input_data = predefined_samples[selected_sample_key]
            st.caption(f"Selected sample for: **{selected_sample_key.replace('Sample: ', '')}**")
            # Prepare data for prediction
            if isinstance(sample_input_data, np.ndarray):
                data_to_predict = sample_input_data
                data_source_key = selected_sample_key
                display_mode = "single"
                # Ensure it's 2D for scaler later
                if data_to_predict.ndim == 1:
                     data_to_predict = data_to_predict.reshape(1, -1)
            else:
                st.error(f"Predefined sample '{selected_sample_key}' is not in the expected format (Numpy array).")
        else:
            st.info("Select a sample from the dropdown above.")


# --- File Upload Logic (Main Page) ---
elif input_method == "Upload CSV File":
    uploaded_file = st.file_uploader(
        f"Upload CSV (must include {N_SELECTED_FEATURES} required features; 'Activity' column optional for comparison)",
        type=["csv"]
    )
    if uploaded_file is not None:
        try:
            input_df_from_file = pd.read_csv(uploaded_file)
            st.write("Uploaded DataFrame Preview (first 5 rows):")
            st.dataframe(input_df_from_file.head())

            input_df_features_only = None # Reset

            # --- Extract True Labels (if available) ---
            if 'Activity' in input_df_from_file.columns:
                true_labels = input_df_from_file['Activity'].copy() # Extract true labels
                # Prepare DataFrame with features only for validation/prediction
                try:
                    input_df_features_only = input_df_from_file.drop('Activity', axis=1)
                    st.info("Found 'Activity' column. Will show alongside predictions.")
                except KeyError: # Should not happen due to 'in' check, but good practice
                     st.warning("Could not remove 'Activity' column though it seemed present.")
                     input_df_features_only = input_df_from_file.copy()
                     true_labels = None # Invalidate true labels if drop failed
            else:
                input_df_features_only = input_df_from_file.copy() # Use all columns as features
                st.warning("Optional 'Activity' column not found in uploaded file. Cannot show true labels for comparison.")
                true_labels = None # Explicitly set to None

            # --- Validation (on feature columns only) ---
            if input_df_features_only is not None:
                missing_cols = set(final_selected_columns) - set(input_df_features_only.columns)
                extra_cols = set(input_df_features_only.columns) - set(final_selected_columns)

                if missing_cols:
                    validation_error = f"Error: Uploaded CSV is missing required feature columns: {', '.join(missing_cols)}"
                    st.error(validation_error)
                elif extra_cols:
                     validation_warning = f"Warning: Uploaded CSV contains extra columns not used for prediction: {', '.join(extra_cols)}. These will be ignored."
                     st.warning(validation_warning)
                     # Select only the required columns and keep their order
                     input_df_features_only = input_df_features_only[final_selected_columns]
                     st.success(f"CSV structure validated ({input_df_features_only.shape[0]} rows). Required columns found. Ready for prediction.")
                     upload_success = True
                else:
                    # If no missing and no extra columns, just ensure correct order
                    input_df_features_only = input_df_features_only[final_selected_columns]
                    st.success(f"CSV structure validated ({input_df_features_only.shape[0]} rows). All required columns present. Ready for prediction.")
                    upload_success = True

                # If validation passed, prepare data for prediction
                if upload_success:
                    data_to_predict = input_df_features_only # Use the validated & ordered features DataFrame
                    data_source_key = "Uploaded File"
                    display_mode = "multi"
                    # 'true_labels' variable was set earlier during file processing

        except pd.errors.EmptyDataError:
            validation_error = "Error: The uploaded CSV file is empty."
            st.error(validation_error)
            data_to_predict = None
            upload_success = False
        except Exception as e:
            validation_error = f"Error processing CSV file: {e}"
            st.error(validation_error)
            data_to_predict = None
            upload_success = False
    else:
        st.info(f"Upload a CSV file containing the required {N_SELECTED_FEATURES} features.")


st.divider()

# --- Prediction Area ---
st.subheader("Run Prediction")

# Only show button if we have valid data to predict from either method
if data_to_predict is not None and data_source_key is not None:
    if st.button(f"Predict Activity for {data_source_key}", key="predict_button"):
        try:
            # --- Preprocessing ---
            # Ensure data is numpy array for scaler
            if isinstance(data_to_predict, pd.DataFrame):
                 data_to_predict_prepared = data_to_predict.values
            elif isinstance(data_to_predict, np.ndarray):
                 # Already checked shape for sample, assume DataFrame from upload gave 2D
                 data_to_predict_prepared = data_to_predict
            else:
                 st.error("Input data is not in a recognizable format (DataFrame or Numpy Array).")
                 st.stop() # Stop if data format is wrong before scaling

            # Scaling
            scaled_data = scaler.transform(data_to_predict_prepared)
            num_samples = scaled_data.shape[0]

            # --- Prediction ---
            predictions_encoded = None
            with st.spinner(f"Predicting using {selected_model_name} for {num_samples} sample(s)..."):
                if selected_model_name == "Logistic Regression":
                    predictions_encoded = log_reg_model.predict(scaled_data)
                elif selected_model_name == "Random Forest":
                    predictions_encoded = rf_model.predict(scaled_data)
                elif selected_model_name == "LSTM":
                    # Reshape data for LSTM: (n_samples, n_timesteps, n_features)
                    # Assuming each row is a single time step
                    reshaped_data = scaled_data.reshape((num_samples, 1, N_SELECTED_FEATURES))
                    prediction_probs = lstm_model.predict(reshaped_data)
                    # Get the index of the highest probability for each sample
                    predictions_encoded = np.argmax(prediction_probs, axis=1)
                else:
                    st.error(f"Model '{selected_model_name}' prediction logic not implemented.")


            # --- Decode and Display Results ---
            if predictions_encoded is not None:
                predicted_activities = label_encoder.inverse_transform(predictions_encoded)
                st.subheader("Prediction Results")

                if display_mode == "single":
                    # Display single result with image
                    activity_name = predicted_activities[0]
                    st.success(f"The predicted activity is: **{activity_name}**")

                    # Attempt to display image for the activity
                    image_filename = f"{activity_name}.jpeg" # Assumes image names match activity names
                    image_path = os.path.join(IMAGES_DIR_ABS, image_filename) # Use absolute path

                    col1_res, col2_res = st.columns([0.7, 0.3]) # Adjust column ratios if needed
                    with col2_res: # Display image in the smaller right column
                         if os.path.exists(image_path):
                             st.image(image_path, caption=f"Activity: {activity_name}", width=150) # Adjust width as needed
                         else:
                             st.warning(f"Image not found for {activity_name} (Expected: {image_filename} in {IMAGES_DIR})")

                else: # display_mode == "multi" (Uploaded CSV)
                    # Create results DataFrame with predicted activity
                    results_df = pd.DataFrame({'Predicted_Activity': predicted_activities})

                    # Add True Activity if available from upload
                    if true_labels is not None:
                        # Ensure true_labels Series aligns with results_df index
                        results_df['True_Activity'] = true_labels.reset_index(drop=True)
                        # Reorder columns for clarity
                        results_df = results_df[['True_Activity', 'Predicted_Activity']]
                        # Calculate accuracy
                        try:
                           accuracy = np.mean(results_df['True_Activity'] == results_df['Predicted_Activity']) * 100
                           st.metric("Prediction Accuracy on Uploaded Data", f"{accuracy:.2f}%")
                        except Exception as acc_e:
                           st.warning(f"Could not calculate accuracy: {acc_e}")


                    # Combine with first few input features for context
                    # Use the original DataFrame before .values conversion if possible
                    # data_to_predict should be the features DataFrame here
                    if isinstance(data_to_predict, pd.DataFrame):
                        input_features_df = data_to_predict.reset_index(drop=True)
                    else: # Should not happen based on logic, but fallback
                        input_features_df = pd.DataFrame(data_to_predict_prepared, columns=final_selected_columns)

                    # Display N first features + results
                    N_FEATURES_TO_SHOW = 5
                    features_subset = input_features_df.iloc[:, :N_FEATURES_TO_SHOW]
                    display_df = pd.concat([features_subset, results_df], axis=1)

                    st.dataframe(display_df)

                    # Provide download option for results (now includes True_Activity if present)
                    csv = display_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name=f'{selected_model_name.lower().replace(" ", "_")}_predictions.csv',
                        mime='text/csv',
                        key='download_csv_button'
                    )
            else:
                st.error("Could not generate predictions. Check model selection and input data.")
        except Exception as pred_e:
            st.error(f"An error occurred during prediction: {pred_e}")
            import traceback
            st.error(traceback.format_exc()) # Show detailed traceback for debugging

else:
    # Message shown when no valid data is ready yet
    if input_method == "Use Predefined Sample":
         if not predefined_samples:
              pass # Warning already shown
         elif selected_sample_key == "--- Select Sample ---":
              st.info("Select a predefined sample from the dropdown above to enable prediction.")
         # If a sample is selected, data_to_predict should be set, so this else shouldn't trigger
    elif input_method == "Upload CSV File" and not uploaded_file:
        st.info("Upload a CSV file above to enable prediction.")
    elif input_method == "Upload CSV File" and uploaded_file and not upload_success:
        st.warning("Please resolve the CSV validation errors shown above before predicting.")


# Footer Info
st.divider()
st.caption(f"App requires {N_SELECTED_FEATURES} features. Column order in uploaded CSV doesn't matter if names match the training columns: {', '.join(final_selected_columns[:3])}..., etc. The optional 'Activity' column in the CSV allows for accuracy calculation.")
st.caption(f"Ensure '{ARTIFACTS_DIR}' and '{IMAGES_DIR}' directories are present with necessary files.")