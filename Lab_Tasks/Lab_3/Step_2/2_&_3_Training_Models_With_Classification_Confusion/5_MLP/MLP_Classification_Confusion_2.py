import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Selected_Features.csv")  # Adjust path if needed

# Split features and labels
X = df.drop(columns='Class')
y = df['Class']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# MLPClassifier with increased iterations
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, activation='relu', solver='adam', random_state=42)
model.fit(X_train, y_train)

# Predictions on test data
y_pred = model.predict(X_test)

# Accuracy on test data
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Train-Test Split Accuracy: {test_accuracy * 100:.2f}%")

# Generate the classification report WITHOUT specifying zero_division (defaults to 'warn')
report = classification_report(y_test, y_pred)
print("\nClassification Report:\n")
print(report)

# Generate the Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Numerical):\n")
print(conf_matrix)

# Plotting the Confusion Matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for MLP Classifier Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
