import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the Testing Data
test_df = pd.read_csv('Test_Data_1000.csv')

# Handle any NaN or Infinite values before proceeding
test_df = test_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.fillna(test_df.mean())

# Generate Composite Features (These operations are based on your previous code)

# RMS Calculation for some paired columns
test_df['TFE Motor speed_and_TFE Steam pressure PV_RMS'] = np.sqrt(
    (test_df['TFE Motor speed']**2 + test_df['TFE Steam pressure PV']**2) / 2
)

# Addition of columns
test_df['TFE Motor speed_plus_TFE Vacuum pressure PV'] = test_df['TFE Motor speed'] + test_df['TFE Vacuum pressure PV']

# Multiplication of columns
test_df['TFE Motor speed_multipliedby_TFE Steam pressure PV'] = test_df['TFE Motor speed'] * test_df['TFE Steam pressure PV']

# Division of columns (Avoiding division by zero)
test_df['FFTE Discharge solids_dividedby_FFTE Production solids PV'] = test_df['FFTE Discharge solids'] / (test_df['FFTE Production solids PV'] + 1e-6)
test_df['TFE Motor speed_dividedby_TFE Steam pressure PV'] = test_df['TFE Motor speed'] / (test_df['TFE Steam pressure PV'] + 1e-6)
test_df['TFE Motor speed_dividedby_TFE Vacuum pressure PV'] = test_df['TFE Motor speed'] / (test_df['TFE Vacuum pressure PV'] + 1e-6)

# Additional Composite Features
test_df['FFTE Discharge solids_dividedby_TFE Production solids PV'] = test_df['FFTE Discharge solids'] / (test_df['TFE Production solids PV'] + 1e-6)
test_df['FFTE Feed tank level PV_dividedby_FFTE Heat temperature 3'] = test_df['FFTE Feed tank level PV'] / (test_df['FFTE Heat temperature 3'] + 1e-6)
test_df['FFTE Heat temperature 1_dividedby_FFTE Temperature 3 - 1'] = test_df['FFTE Heat temperature 1'] / (test_df['FFTE Temperature 3 - 1'] + 1e-6)

# Load the Processed Training Data to ensure column match
train_df = pd.read_csv('Selected_Features.csv')

# Extract the feature columns used in the Training Data (excluding 'Class')
feature_columns = train_df.columns.drop('Class')

# Check for missing columns
missing_columns = set(feature_columns) - set(test_df.columns)
if missing_columns:
    print(f"Warning: The following columns are still missing in the testing data: {missing_columns}")
else:
    print("All required columns are now present in the testing data.")

# Select only the columns present in the training data (in the same order)
test_features = test_df[feature_columns]

# Apply Standardization
scaler = StandardScaler()
test_features_scaled = scaler.fit_transform(test_features)

# Convert the scaled features into a DataFrame with the same feature names
test_features_df = pd.DataFrame(test_features_scaled, columns=feature_columns)

# Add the 'Class' column from the original testing data
test_features_df['Class'] = test_df['Class']

# Save the processed testing data for further use
test_features_df.to_csv('Processed_Test_Data.csv', index=False)
print("\nTesting data has been successfully processed and saved as 'Processed_Test_Data.csv'.")
