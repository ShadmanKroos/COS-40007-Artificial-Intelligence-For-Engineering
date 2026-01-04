import pandas as pd

# Load the datasets
boning_df = pd.read_csv("Boning.csv")
slicing_df = pd.read_csv("Slicing.csv")

# Define the columns to extract
columns_needed = [
    "Frame",
    "Right Toe x", "Right Toe y", "Right Toe z",
    "Left Toe x", "Left Toe y", "Left Toe z"
]

# Extract relevant columns
boning_selected = boning_df[columns_needed].copy()
slicing_selected = slicing_df[columns_needed].copy()

# Adjust slicing frame numbers 
max_frame_boning = boning_selected["Frame"].max()
slicing_selected["Frame"] += max_frame_boning + 1

# Add class labels
boning_selected["Class"] = 0  # Boning
slicing_selected["Class"] = 1  # Slicing

# Combine datasets
combined_df = pd.concat([boning_selected, slicing_selected], ignore_index=True)

# Print first few rows 
print(combined_df.head().to_string(index=False))

# Save to CSV 
combined_df.to_csv("combined_boning_slicing_data.csv", index=False)
