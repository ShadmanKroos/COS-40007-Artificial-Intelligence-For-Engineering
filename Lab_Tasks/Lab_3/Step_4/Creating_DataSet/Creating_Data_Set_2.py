import pandas as pd

# Load the original data set
input_file = 'Selected_Features.csv'
df = pd.read_csv(input_file)

# Extract columns that end with 'SP' or 'PV'
filtered_columns = [col for col in df.columns if col.endswith('SP') or col.endswith('PV')]

# Add the 'Class' column to the filtered list if it's present in the original dataframe
if 'Class' in df.columns:
    filtered_columns.append('Class')
else:
    raise ValueError("The dataset must contain a 'Class' column.")

# Filter the dataset with the selected columns
filtered_df = df[filtered_columns]

# Display the number of columns that match the criteria
print(f"Number of columns ending with 'SP' or 'PV': {len(filtered_columns)-1} (excluding 'Class')")

# Save the filtered data to a new CSV file
output_file = 'Filtered_SP_PV_Features_with_Class.csv'
filtered_df.to_csv(output_file, index=False)

print(f"Filtered data has been saved as '{output_file}'")
