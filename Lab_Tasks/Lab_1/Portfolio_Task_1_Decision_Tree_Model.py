import pandas as pd
import numpy as np
from sklearn. model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataframe = pd.read_csv("wdbc_feature_engineered.csv")

#Repair NaN and Infinite Values
print("Checking for infinite values:\n")
print(dataframe.replace([np.inf, -np.inf], np.nan).isnull().sum())
dataframe.replace([np.inf, -np.inf],np.nan, inplace = True)

print("Checking NaN values:\n")
print(dataframe.isnull().sum())

dataframe.fillna(dataframe.mean(), inplace = True)

fixed_file = "wdbc_feature_engineering_cleaned.csv"
dataframe.to_csv(fixed_file, index = False)
print("Saved file: ", fixed_file)

#Declare Target
target = "Diagnosis"

#Declare Features
feature_set = {
    "Set 1 - Basic Features": ["Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean"],
    "Set 2 - Compactness & Concavity": ["Compactness_mean", "Concavity_mean", "Concave_points_mean", "Smoothness_mean"],
    "Set 3 - Aspect & Symmetry Ratios": ["Aspect_Ratio", "Symmetrical_Ratio", "MinWorst_Difference"],
    "Set 4 - Polynomial Interactions": ["Compactness_mean Concavity_mean", "Concavity_mean Concave_points_mean", 
                                        "Compactness_mean Smoothness_mean"],
    "Set 5 - Combined Important Features": ["Radius_mean", "Compactness_mean", "Concavity_mean", 
                                            "Concave_points_mean", "Symmetrical_Ratio"]
}

#Splitting Data (70% trained, 30% test)
training_dataframe, testing_dataframe = train_test_split (dataframe, test_size = 0.3, random_state = 42)
final_accuracy = {}


#Training the Decision Tree
for set_name, features in feature_set.items():
    x_train, x_test = training_dataframe[features], testing_dataframe[features]
    y_train, y_test = training_dataframe[target], testing_dataframe[target]

    tree_model = DecisionTreeClassifier(random_state= 42)
    tree_model.fit(x_train, y_train) #train the model

    y_prediction = tree_model.predict(x_test) #prediction

    accuracy_rate = accuracy_score(y_test, y_prediction)

    final_accuracy[set_name] = accuracy_rate

print("\nAccuracy of Decision Tree Models using Different Feature Sets:")
for set_name, acc in final_accuracy.items():
    print(f"{set_name}: {acc:.4f}")    


