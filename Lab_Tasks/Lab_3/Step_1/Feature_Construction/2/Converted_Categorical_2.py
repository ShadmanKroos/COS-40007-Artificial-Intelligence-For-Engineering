import pandas as pd

# Load the dataset (Constant_Removed.csv)
df = pd.read_csv('Constant_Removed.csv')

# Display original data types before conversion
print("\nData Types Before Conversion:\n")
print(df.dtypes)

# Set a threshold for the maximum number of unique values to be considered "few"
unique_threshold = 10

# Identify columns to convert, ignoring the 'Class' column
columns_to_convert = [col for col in df.columns if col != 'Class' and 
                      df[col].dtype in ['int64', 'float64'] and 
                      df[col].nunique() <= unique_threshold]

# Display the columns identified for conversion
if columns_to_convert:
    print("\nColumns to be converted to categorical features (Excluding 'Class'):", columns_to_convert)
    
    # Convert identified columns to categorical
    for col in columns_to_convert:
        df[col] = df[col].astype('category')
    
    print("\nColumns have been successfully converted to categorical type.")
else:
    print("No columns with few integer values found for conversion.")

# Display data types after conversion
print("\nData Types After Conversion:\n")
print(df.dtypes)

# Save the modified DataFrame as 'Converted_Categorical.csv'
df.to_csv('Converted_Categorical.csv', index=False)
print("\nModified dataset saved as 'Converted_Categorical.csv' successfully.")
