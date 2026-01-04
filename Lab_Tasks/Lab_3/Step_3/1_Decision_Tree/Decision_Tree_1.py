import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

# Load the trained Decision Tree model
filename = 'Decision_Tree_Model.pkl'
model = pickle.load(open(filename, 'rb'))

# Load the processed testing dataset
df_test = pd.read_csv('Processed_Test_Data_matched.csv')

# Separate features (X) and target (y)
X_test = df_test.drop(columns=['Class'])
y_test = df_test['Class']

# Make predictions using the loaded model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")
