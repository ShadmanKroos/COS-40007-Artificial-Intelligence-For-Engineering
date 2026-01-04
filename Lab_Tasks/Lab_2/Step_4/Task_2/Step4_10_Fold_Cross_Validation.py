import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv("Step3_Preprocessing_Compution.csv")

# Separate features and label
X = df.drop("Class", axis=1)
y = df["Class"]

# Create SVM classifier with RBF kernel
clf = svm.SVC(kernel='rbf')

# Perform 10-fold cross-validation
scores = cross_val_score(clf, X, y, cv=10)

# Print each fold's accuracy and mean accuracy
print("10-Fold Cross-Validation Accuracies:")
for i, score in enumerate(scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")

print(f"\nAverage Accuracy: {scores.mean() * 100:.2f}%")
