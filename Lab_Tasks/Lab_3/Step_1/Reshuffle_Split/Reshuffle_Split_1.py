import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('vegemite.csv')

# Shuffle the dataset (With random_state for reproducibility)
shuffled_df = df.sample(n=len(df), random_state=42)
shuffled_df = shuffled_df.reset_index(drop=True)

# Display the class distribution (Use 'Class' instead of 'class')
print("Original Class Distribution:")
print(shuffled_df['Class'].value_counts())

# Ensure equal distribution of classes in the test dataset
test_data = pd.concat([
    shuffled_df[shuffled_df['Class'] == label].sample(n=300, random_state=42)
    for label in shuffled_df['Class'].unique()
])

# Remove the test data from the original dataframe to create the training dataset
train_data = shuffled_df.drop(test_data.index)

# Save the test and training datasets to CSV files
test_data.to_csv('Test_Data_1000.csv', index=False)
train_data.to_csv('Train_Data_14000.csv', index=False)

# Display the first few rows of each file
print("\nFirst 5 rows of Test_Data_1000.csv:")
print(test_data.head())

print("\nFirst 5 rows of Train_Data_14000.csv:")
print(train_data.head())
