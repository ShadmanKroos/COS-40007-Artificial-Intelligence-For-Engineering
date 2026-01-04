import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle  # Importing pickle for saving the model

# Load the dataset
df = pd.read_csv('Selected_Features.csv')

# Separate features and class
X = df.drop(columns=['Class'])
y = df['Class']

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Create SGD Classifier Model
model = SGDClassifier(random_state=42, class_weight='balanced', max_iter=1000, tol=1e-3)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of the SGD Model (Scaled Data): {accuracy * 100:.2f}%")

# Generate Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Generate Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plotting the Confusion Matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for SGD Model (Scaled Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Save the trained model to a file using pickle
filename = 'SGD_Model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"\nModel saved successfully as '{filename}'.")
