import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the step 3 dataset
df = pd.read_csv("Step3_Preprocessing_Compution.csv")

# Separate features and label
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM classifier with RBF kernel
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Compute and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Train-Test Split SVM Accuracy (RBF Kernel): {accuracy * 100:.2f}%")
