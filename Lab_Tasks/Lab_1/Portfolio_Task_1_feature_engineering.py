import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

#name of the files
csv_file = "wdbc.csv"
data_file = "wdbc.data"
normalized_data_file = "wdbc_normalized.csv"
feature_engineered_file = "wdbc_feature_engineered.csv"

#columns
columns_all = [
    "ID", "Diagnosis", "Radius_mean", "Texture_mean", "Perimeter_mean", "Area_mean",
    "Smoothness_mean", "Compactness_mean", "Concavity_mean", "Concave_points_mean", "Symmetry_mean",
    "Fractal_dimension_mean", "Radius_se", "Texture_se", "Perimeter_se", "Area_se", "Smoothness_se",
    "Compactness_se", "Concavity_se", "Concave_points_se", "Symmetry_se", "Fractal_dimension_se",
    "Radius_worst", "Texture_worst", "Perimeter_worst", "Area_worst", "Smoothness_worst",
    "Compactness_worst", "Concavity_worst", "Concave_points_worst", "Symmetry_worst", "Fractal_dimension_worst"
]

#read data
dataframe = pd.read_csv (data_file, header = None, names = columns_all)

#Convert M > 1 and B > 0 in "Diagnosis"
dataframe['Diagnosis'] = dataframe['Diagnosis'].map({'M':1, 'B':0})

dataframe.to_csv(csv_file, index = False)
print("\n Data of first 5 rows before normalization")
print(dataframe.head())

#Normalizing
normalizer = MinMaxScaler()
excluded_features = dataframe.columns.difference(['ID', 'Diagnosis']) #exclude ID and Diagnosis

dataframe_normalized = dataframe.copy()
dataframe_normalized[excluded_features] = normalizer.fit_transform(dataframe[excluded_features]) #Assgining Normalized Values in the default columns [Default Columns = Columns before normalization]
dataframe_normalized['Diagnosis'] = dataframe_normalized['Diagnosis'].astype(int)

dataframe.to_csv(normalized_data_file, index = False)
print("\n Normalized Data for First 5 Rows")
print(dataframe_normalized.head())

#feature_engineering
dataframe_feature_engineered = dataframe_normalized.copy()

#Creating_Features
dataframe_feature_engineered["Aspect_Ratio"] = dataframe_feature_engineered["Perimeter_mean"] / dataframe_feature_engineered["Radius_mean"]
dataframe_feature_engineered["Compactness_Ratio"] = dataframe_feature_engineered["Compactness_mean"] * dataframe_feature_engineered["Smoothness_mean"]
dataframe_feature_engineered["Concavity_Severeness"] = dataframe_feature_engineered["Concavity_mean"] * dataframe_feature_engineered["Concave_points_mean"]
dataframe_feature_engineered["Symmetrical_Ratio"] = dataframe_feature_engineered["Symmetry_mean"] / dataframe_feature_engineered["Fractal_dimension_mean"]
dataframe_feature_engineered["MinWorst_Difference"] = dataframe_feature_engineered["Radius_worst"] - dataframe_feature_engineered["Radius_mean"]

#Polynomial Features

polynomial_features = ["Compactness_mean", "Concavity_mean", "Concave_points_mean", "Smoothness_mean"]
poly_info = PolynomialFeatures(degree = 2, interaction_only = True, include_bias = False)
poly_transformation = poly_info.fit_transform(dataframe_feature_engineered[polynomial_features])

#Polynomial_Dataframe
feature_names_polynomial = poly_info.get_feature_names_out(polynomial_features)
polynomial_dataframe = pd.DataFrame(poly_transformation, columns = feature_names_polynomial)

#Attach with the dataframe
dataframe_attach_polynomial = pd.concat([dataframe_feature_engineered, polynomial_dataframe], axis = 1)
dataframe_attach_polynomial.to_csv(feature_engineered_file, index = False)


print("\nData of first 5 rows after feature engineering")
print(dataframe_attach_polynomial.head())
print(f"\Dataset contains {dataframe_attach_polynomial.shape[0]} rows and {dataframe_attach_polynomial.shape[1]} columns")






