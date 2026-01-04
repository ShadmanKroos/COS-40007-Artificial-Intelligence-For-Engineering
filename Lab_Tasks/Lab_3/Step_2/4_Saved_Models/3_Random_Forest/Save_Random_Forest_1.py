import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Importing pickle for saving the model

# Load the dataset
df = pd.read_csv('Selected_Features.csv')

# Separate features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Split the dataset into training set and test set (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Random Forest Classifier Model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the Random Forest Model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the Random Forest Model: {accuracy * 100:.2f}%")

# Generate Classification Report
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
plt.title('Confusion Matrix for Random Forest Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the trained model to a file using pickle
filename = 'Random_Forest_Model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"\nModel saved successfully as '{filename}'.")
