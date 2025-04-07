import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

# --- Configuration (Adjust paths if needed) ---
DATA_PATH = 'test.csv' # Path to test data CSV (MUST contain 'Activity' column)
ARTIFACTS_DIR = "saved_artifacts"
FINAL_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "final_columns.joblib")
# Changed output filename to reflect content
OUTPUT_CSV_PATH = "har_test_sample_10_stratified_with_labels.csv"
OUTPUT_CSV_PATH1 = "har_test_sample_10_stratified.csv"
NUM_SAMPLES = 10
RANDOM_STATE = 42

print("Generating stratified test sample CSV (features + labels) from UCI HAR dataset...")

try:
    # 1. Load selected feature names
    final_selected_columns = joblib.load(FINAL_COLUMNS_PATH)
    print(f"Loaded {len(final_selected_columns)} selected feature names.")

    # 2. Load the original test dataset (including 'Activity')
    full_test_df = pd.read_csv(DATA_PATH)
    print(f"Loaded original test data from: {DATA_PATH}")

    # 3. Check for 'Activity' column and separate features/target
    if 'Activity' not in full_test_df.columns:
        print(f"Error: 'Activity' column not found in {DATA_PATH}. Cannot perform stratified sampling or include true labels.")
    else:
        y_test_labels = full_test_df['Activity']
        X_test_full = full_test_df.drop('Activity', axis=1, errors='ignore')

        # 4. Select only the required feature columns
        missing_cols = set(final_selected_columns) - set(X_test_full.columns)
        if missing_cols:
            print(f"Error: Required feature columns not found in {DATA_PATH}: {missing_cols}")
        else:
            X_test_selected_features = X_test_full[final_selected_columns]
            print(f"Selected the {len(final_selected_columns)} required features.")

            # 5. Perform Stratified Sampling
            print(f"Performing stratified sampling to get {NUM_SAMPLES} samples...")
            X_sample, _, y_sample, _ = train_test_split(
                X_test_selected_features,
                y_test_labels,
                train_size=NUM_SAMPLES,
                stratify=y_test_labels,
                random_state=RANDOM_STATE
            )
            print(f"Selected {len(X_sample)} feature samples.")
            print(f"Selected {len(y_sample)} corresponding labels.")

            # 6. **Combine features (X_sample) and labels (y_sample) into one DataFrame**
            output_df = X_sample.reset_index(drop=True) # Use reset_index for clean assignment
            output_df.to_csv(OUTPUT_CSV_PATH1, index=False)
            print(f"Successfully saved {len(output_df)} stratified samples (features) with headers to: {OUTPUT_CSV_PATH1}")
            # Add the labels Series as a new column named 'Activity'
            output_df['Activity'] = y_sample.reset_index(drop=True)
            print("Combined features and original activity labels for the sample.")

            # 7. **Save the combined DataFrame to CSV**
            output_df.to_csv(OUTPUT_CSV_PATH, index=False)
            print(f"Successfully saved {len(output_df)} stratified samples (features + Activity label) with headers to: {OUTPUT_CSV_PATH}")

# ... (rest of the error handling remains the same) ...
except FileNotFoundError as e:
    print(f"Error: Could not find necessary file: {e}")
    # ... other except blocks ...