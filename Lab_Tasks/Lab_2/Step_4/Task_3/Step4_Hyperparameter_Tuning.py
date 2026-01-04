import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Step3_Preprocessing_Compution.csv")

# Separate features and label
X = df.drop("Class", axis=1)
y = df["Class"]

# Define hyperparameter grid (RBF kernel)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Grid search with verbosity to show progress
grid = GridSearchCV(svm.SVC(), param_grid, cv=5, verbose=1)
grid.fit(X, y)

# Best parameters from grid search
best_params = grid.best_params_
print(f"\nBest Hyperparameters: {best_params}")

# Train-Test Split Accuracy (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_split = svm.SVC(**best_params)
clf_split.fit(X_train, y_train)
y_pred = clf_split.predict(X_test)
split_accuracy = accuracy_score(y_test, y_pred)
print(f"\nTrain-Test Split Accuracy (Tuned SVM): {split_accuracy * 100:.2f}%")

#10-Fold Cross-Validation Accuracy
clf_cv = svm.SVC(**best_params)
cv_scores = cross_val_score(clf_cv, X, y, cv=10)
print("\n10-Fold Cross-Validation Accuracies (Tuned SVM):")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")
print(f"\nAverage CV Accuracy: {cv_scores.mean() * 100:.2f}%")
