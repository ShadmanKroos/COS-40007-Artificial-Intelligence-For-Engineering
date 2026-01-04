import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Load the filtered dataset
input_file = 'Filtered_SP_PV_Features_with_Class.csv'
df = pd.read_csv(input_file)

# Select only columns that end with 'SP'
sp_columns = [col for col in df.columns if col.endswith('SP')]

# Check if Class column is present
if 'Class' not in df.columns:
    raise ValueError("The dataset must contain a 'Class' column.")

# Extract features (X) and labels (y)
X = df[sp_columns]
y = df['Class']

# Split the data into training and testing sets (70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")

# Print the Decision Tree using export_text
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
