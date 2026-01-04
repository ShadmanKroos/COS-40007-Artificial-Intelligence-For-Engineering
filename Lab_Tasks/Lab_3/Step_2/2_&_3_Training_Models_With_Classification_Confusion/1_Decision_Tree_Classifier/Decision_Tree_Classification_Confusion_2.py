import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Selected_Features.csv')

# Separate features (X) and target (y)
X = df.drop(columns=['Class'])
y = df['Class']

# Split the dataset into training set and test set (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Decision Tree classifier object with random_state=42
model = DecisionTreeClassifier(random_state=42)

# Train Decision Tree Classifier
model = model.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy (In Percentage)
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy: {accuracy:.2f}%")

# Generate Classification Report WITHOUT specifying zero_division (defaults to 'warn')
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n")
print(report)

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the Confusion Matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Decision Tree Model (Selected Features Dataset)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
