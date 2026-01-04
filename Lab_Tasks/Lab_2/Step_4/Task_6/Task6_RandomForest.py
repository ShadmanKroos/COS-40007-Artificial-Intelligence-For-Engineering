import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# Load Step 1 dataset
df = pd.read_csv("combined_boning_slicing_data.csv")

# Drop Frame column, separate features and target
X = df.drop(["Frame", "Class"], axis=1)
y = df["Class"]

# Train-Test Split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_split = RandomForestClassifier(random_state=42)
clf_split.fit(X_train, y_train)
y_pred = clf_split.predict(X_test)
accuracy_split = accuracy_score(y_test, y_pred)
print(f"Train-Test Split Accuracy (Random Forest): {accuracy_split * 100:.2f}%")

# 10-Fold Cross Validation
clf_cv = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(clf_cv, X, y, cv=10)
print("\n10-Fold Cross-Validation Accuracies (Random Forest):")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")
print(f"\nAverage CV Accuracy: {cv_scores.mean() * 100:.2f}%")
