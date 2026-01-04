import pandas as pd

# Load the original dataset (without modifying it directly)
df = pd.read_csv('Train_Data_14000.csv')

# Find constant columns (all values are the same)
constant_columns = [col for col in df.columns if df[col].nunique() == 1]

# Display constant columns, if any
if constant_columns:
    print("Constant Columns Found:", constant_columns)
    # Remove constant columns from the dataset
    df_modified = df.drop(columns=constant_columns)
    print("\nConstant columns have been removed successfully.")
else:
    print("No constant columns found.")
    # No columns removed, so the modified dataset is the same as the original
    df_modified = df.copy()

# Save the modified DataFrame as 'Constant_Removed.csv' without modifying the original file
df_modified.to_csv('Constant_Removed.csv', index=False)
print("\nModified dataset saved as 'Constant_Removed.csv' successfully.")
