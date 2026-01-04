import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Step3_Preprocessing_Compution.csv")

# Separate features and label
X = df.drop("Class", axis=1)
y = df["Class"]

# Apply PCA to reduce to 10 principal components
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

# Use best parameters from GridSearchCV
best_params = {
    'C': 0.1,
    'gamma': 0.001,
    'kernel': 'rbf'
}

# 70/30 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
clf_split = SVC(**best_params)
clf_split.fit(X_train, y_train)
y_pred = clf_split.predict(X_test)
accuracy_split = accuracy_score(y_test, y_pred)
print(f"Train-Test Split Accuracy (10 PCA Components, Tuned SVM): {accuracy_split * 100:.2f}%")

# 10-Fold Cross Validation
clf_cv = SVC(**best_params)
cv_scores = cross_val_score(clf_cv, X_pca, y, cv=10)
print("\n10-Fold Cross-Validation Accuracies (10 PCA Components, Tuned SVM):")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score * 100:.2f}%")
print(f"\nAverage CV Accuracy: {cv_scores.mean() * 100:.2f}%")
