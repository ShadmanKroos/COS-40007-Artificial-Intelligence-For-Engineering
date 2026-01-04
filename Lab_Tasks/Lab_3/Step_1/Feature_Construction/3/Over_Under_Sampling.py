import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# Load the dataset (Converted_Categorical.csv)
df = pd.read_csv('Converted_Categorical.csv')

# Check the initial class distribution
class_counts = Counter(df['Class'])
print("\nOriginal Class Distribution:\n", class_counts)

# Splitting the data into features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Step 1: Oversampling with SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled_smote, y_resampled_smote = smote.fit_resample(X, y)

# Display the class distribution after SMOTE
print("\nClass Distribution After Oversampling (SMOTE):")
print(Counter(y_resampled_smote))

# Step 2: Undersampling with Tomek Links (On SMOTE Data)
tomek_links = TomekLinks()
X_resampled_final, y_resampled_final = tomek_links.fit_resample(X_resampled_smote, y_resampled_smote)

# Display the class distribution after Tomek Links
print("\nClass Distribution After Undersampling (Tomek Links):")
print(Counter(y_resampled_final))

# Combine the final resampled data into a DataFrame
df_final = pd.concat([X_resampled_final, y_resampled_final], axis=1)

# Save the final balanced dataset as 'Over_Under_Sampling.csv'
df_final.to_csv('Over_Under_Sampling.csv', index=False)
print("\nFinal dataset saved as 'Over_Under_Sampling.csv' successfully.")
