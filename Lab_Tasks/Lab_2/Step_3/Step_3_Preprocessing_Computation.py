import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import auc

# Load input file
df = pd.read_csv("Step2_Composite_Features.csv")

# Columns 2–19 (index 1 to 18) → sensor features only
feature_cols = df.columns[1:19]

# Result list for processed chunks
feature_rows = []

# Process in 60-row chunks (1 minute)
for start in range(0, len(df), 60):
    chunk = df.iloc[start:start + 60]
    
    # Skip incomplete chunk
    if len(chunk) < 60:
        continue
    
    stats = {}
    
    for col in feature_cols:
        series = chunk[col].values

        stats[f"{col}_mean"] = np.mean(series)
        stats[f"{col}_std"] = np.std(series)
        stats[f"{col}_min"] = np.min(series)
        stats[f"{col}_max"] = np.max(series)
        stats[f"{col}_auc"] = auc(np.arange(len(series)), series)
        stats[f"{col}_peaks"] = len(find_peaks(series)[0])
    
    # set class label using majority vote in the chunk
    stats["Class"] = chunk["Class"].mode()[0]
    
    feature_rows.append(stats)

# Create final DataFrame
summary_df = pd.DataFrame(feature_rows)

# Print preview
print(summary_df.head().to_string(index=False))

# Save to file
summary_df.to_csv("Step3_Preprocessing_Compution.csv", index=False)
