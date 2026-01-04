import pandas as pd
import numpy as np
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Over_Under_Sampling.csv)
df = pd.read_csv('Over_Under_Sampling.csv')

# Display available columns
print("\nAvailable Columns:")
print(df.columns)

# Identify numerical columns only (excluding the 'Class' column)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
numerical_columns = [col for col in numerical_columns if col != 'Class']

# Display numerical columns
print("\nNumerical Columns Identified:")
print(numerical_columns)

# Calculate the correlation matrix using Pearson Correlation Coefficient
correlation_matrix = df[numerical_columns].corr(method='pearson')

# Display the correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plotting heatmap for visualization (Without annotation values)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Setting a dynamic threshold based on the dataset's correlation values
correlations = correlation_matrix.values.flatten()
correlations = correlations[(correlations != 1) & (correlations != -1)]  # Removing perfect correlations

average_correlation = np.mean(np.abs(correlations))
correlation_threshold = max(0.7, average_correlation)  # Setting threshold dynamically based on average correlation

print(f"\nDynamic Correlation Threshold Set to: {correlation_threshold:.2f}")

# Find pairs of columns with high correlation (absolute value)
correlated_pairs = []
for col1, col2 in combinations(numerical_columns, 2):
    correlation_value = abs(correlation_matrix.loc[col1, col2])
    if correlation_value >= correlation_threshold:
        correlated_pairs.append((col1, col2, correlation_value))

# Display the identified correlated pairs
if correlated_pairs:
    print("\nStrongly Correlated Pairs (Correlation >= Dynamic Threshold):")
    for pair in correlated_pairs:
        print(f"{pair[0]} and {pair[1]}: Correlation = {pair[2]:.2f}")
else:
    print("\nNo strongly correlated pairs found with the dynamic threshold.")

# Initialize a dictionary to store all new composite features
new_columns = {}
composite_features = []

# Create composite features for correlated columns only
for col1, col2, correlation_value in correlated_pairs:
    # Creating RMS
    new_col_rms = f'{col1}_{col2}_RMS'
    new_columns[new_col_rms] = np.sqrt((df[col1]**2 + df[col2]**2) / 2)
    composite_features.append(new_col_rms)

    # Addition
    new_col_add = f'{col1}_plus_{col2}'
    new_columns[new_col_add] = df[col1] + df[col2]
    composite_features.append(new_col_add)
    
    # Multiplication
    new_col_mul = f'{col1}_times_{col2}'
    new_columns[new_col_mul] = df[col1] * df[col2]
    composite_features.append(new_col_mul)
    
    # Division (Avoiding division by zero)
    if not (df[col2] == 0).all():
        new_col_div = f'{col1}_dividedby_{col2}'
        new_columns[new_col_div] = df[col1] / df[col2]
        composite_features.append(new_col_div)

# Convert the dictionary to a DataFrame and concatenate it with the original DataFrame
new_columns_df = pd.DataFrame(new_columns)
df = pd.concat([df, new_columns_df], axis=1)

# Save the modified DataFrame with composite features if any were created
if composite_features:
    df.to_csv('Composite_Features.csv', index=False)
    print("\nComposite features created successfully and saved as 'Composite_Features.csv'.")
    print("\nCreated Composite Features:")
    print(composite_features)
else:
    print("\nNo composite features could be generated based on correlation analysis.")

# Count the total number of features in the final dataset (INCLUDING 'Class')
total_features = len(df.columns)
print(f"\nTotal Number of Features in the Final Dataset (Including 'Class'): {total_features}")
