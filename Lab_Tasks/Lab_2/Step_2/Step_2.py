import pandas as pd
import numpy as np

# Load the combined dataset
df = pd.read_csv("combined_boning_slicing_data.csv")

# Right Toe values
x_r = df["Right Toe x"]
y_r = df["Right Toe y"]
z_r = df["Right Toe z"]

# Left Toe values
x_l = df["Left Toe x"]
y_l = df["Left Toe y"]
z_l = df["Left Toe z"]

# Composite Features for Right Toe
df["RToe_rms_xy"] = np.sqrt(x_r**2 + y_r**2)
df["RToe_rms_yz"] = np.sqrt(y_r**2 + z_r**2)
df["RToe_rms_zx"] = np.sqrt(z_r**2 + x_r**2)
df["RToe_rms_xyz"] = np.sqrt(x_r**2 + y_r**2 + z_r**2)
df["RToe_roll"] = np.degrees(np.arctan2(y_r, np.sqrt(x_r**2 + z_r**2)))
df["RToe_pitch"] = np.degrees(np.arctan2(x_r, np.sqrt(y_r**2 + z_r**2)))

# Composite Features for Left Toe
df["LToe_rms_xy"] = np.sqrt(x_l**2 + y_l**2)
df["LToe_rms_yz"] = np.sqrt(y_l**2 + z_l**2)
df["LToe_rms_zx"] = np.sqrt(z_l**2 + x_l**2)
df["LToe_rms_xyz"] = np.sqrt(x_l**2 + y_l**2 + z_l**2)
df["LToe_roll"] = np.degrees(np.arctan2(y_l, np.sqrt(x_l**2 + z_l**2)))
df["LToe_pitch"] = np.degrees(np.arctan2(x_l, np.sqrt(y_l**2 + z_l**2)))

# Reorder columns
ordered_columns = [
    "Frame",
    "Right Toe x", "Right Toe y", "Right Toe z",
    "Left Toe x", "Left Toe y", "Left Toe z",
    "RToe_rms_xy", "RToe_rms_yz", "RToe_rms_zx", "RToe_rms_xyz", "RToe_roll", "RToe_pitch",
    "LToe_rms_xy", "LToe_rms_yz", "LToe_rms_zx", "LToe_rms_xyz", "LToe_roll", "LToe_pitch",
    "Class"
]
df = df[ordered_columns]

# Print first few rows without the index
print(df.head().to_string(index=False))

# Save to new file
df.to_csv("Step2_Composite_Features.csv", index=False)
