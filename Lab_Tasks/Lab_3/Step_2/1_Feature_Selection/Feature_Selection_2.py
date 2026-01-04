import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold

# Load the training dataset
df = pd.read_csv('Composite_Features.csv')

# Display the total number of features before selection
print(f"\nTotal Number of Features Before Selection: {len(df.columns)}")

# Separate features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Step 0: Data Cleaning - Handling Inf and NaN values

# Replace inf values with NaN for easier detection
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Identify columns with NaN values
columns_with_nan = X.columns[X.isna().any()].tolist()
print(f"\nColumns Containing NaN or Inf Values: {columns_with_nan}")

# Option 1: Fill NaN values with the mean of the column
X.fillna(X.mean(), inplace=True)

# Option 2 (Alternative): If a column has too many NaN values, drop it
threshold = 0.7  # Drop column if more than 70% of its values are NaN
columns_to_drop = X.columns[X.isna().mean() > threshold].tolist()
X.drop(columns=columns_to_drop, inplace=True)

print(f"\nNumber of Features After Removing Highly NaN Columns: {len(X.columns)}")

# Step 1: Removing Low-Variance Features
variance_threshold = 0.01  # You can adjust this threshold
selector = VarianceThreshold(threshold=variance_threshold)
X_var = selector.fit_transform(X)

# Keep only the selected columns
selected_columns_var = X.columns[selector.get_support()]
X = X[selected_columns_var]

print(f"\nNumber of Features After Low-Variance Filtering: {len(X.columns)}")

# Step 2: Removing Highly Correlated Features
correlation_matrix = X.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Define a threshold for high correlation
correlation_threshold = 0.85
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
X.drop(columns=to_drop, inplace=True)

print(f"\nNumber of Features After Correlation Filtering: {len(X.columns)}")

# Step 3: Feature Importance using Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Get feature importances and sort them
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
important_features = feature_importances[feature_importances > 0.001]  # Filtering out low importance features
important_features.sort_values(ascending=False, inplace=True)

# Keep only important features
X = X[important_features.index]

print(f"\nNumber of Features After Feature Importance Filtering: {len(X.columns)}")

# Plotting Feature Importance
plt.figure(figsize=(12, 8))
important_features.plot(kind='bar')
plt.title('Feature Importance from Random Forest')
plt.show()

# Combine the reduced features with the target column
final_df = pd.concat([X, y], axis=1)

# Save the final dataset
final_df.to_csv('Selected_Features.csv', index=False)
print("\nFeature selection completed successfully and saved as 'Selected_Features.csv'.")
print(f"\nTotal Number of Features After Selection: {len(final_df.columns)}")
