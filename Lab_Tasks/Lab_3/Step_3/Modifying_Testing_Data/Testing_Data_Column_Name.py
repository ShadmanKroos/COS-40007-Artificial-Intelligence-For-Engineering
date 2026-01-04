import pandas as pd

# Load the training dataset (Selected_Features.csv)
training_data = pd.read_csv('Selected_Features.csv')

# Load the testing dataset (Processed_Test_Data.csv)
testing_data = pd.read_csv('Processed_Test_Data.csv')

# Extract the column names from the training dataset
training_columns = training_data.columns.tolist()

# Check if the testing dataset contains all the training columns
missing_columns = set(training_columns) - set(testing_data.columns)
if missing_columns:
    print(f"Warning: The following columns are missing in the testing data: {missing_columns}")

# Add missing columns to the testing dataset with default values (0 or NaN)
for column in missing_columns:
    testing_data[column] = 0  # Or you can use testing_data[column] = np.nan if you prefer

# Reorder the columns of the testing dataset to match the training dataset
testing_data = testing_data[training_columns]

# Save the processed testing dataset
testing_data.to_csv('Processed_Test_Data_matched.csv', index=False)

print("The processed testing data has been saved as 'Processed_Test_Data_matched.csv'.")
