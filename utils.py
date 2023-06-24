from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans

def transform_to_numerical_columns(df, columns):
  df_to_encode = df[columns]
  one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
  encoded_data = one_hot_encoder.fit_transform(df_to_encode)
  encoded_columns = one_hot_encoder.get_feature_names_out(columns)
  # Create a new DataFrame with the encoded data
  df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)
  # Concatenate the encoded DataFrame with the original DataFrame (excluding the columns to encode)
  df_final = pd.concat([df.drop(columns=columns), df_encoded], axis=1)
  return df_final

def unique_values_in_column(df, column):
  unique_values = df[column].value_counts()
  print(unique_values)

def transform_column_mapping(df, columns_mapping):
  for column, mapping in columns_mapping.items():
    df[column] = df[column].replace(mapping)
    unique_values_in_column(df, column)
  return df

def find_object_columns(df):
  object_columns = df.select_dtypes(include='object')
  object_column_names = object_columns.columns.tolist()
  print(object_column_names)

def check_if_null_or_inf(df, column):
  null_values = df[column].isnull().any()
  inf_values = df[column].isin([float('inf'), float('-inf')]).any()
  print("Null values:", null_values)
  print("Infinite values:", inf_values)

from sklearn.preprocessing import StandardScaler

def standard_scaler(df):
  print(f'the dataframe is of shape{df.shape}')
  numerical_columns = df.select_dtypes(include=['int', 'float']).columns
  print(f'Number of numerical columns {len(numerical_columns.tolist())}')
  scaler = StandardScaler()
  df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
  return df

def find_null_values(df):
  null_values = df.isnull()
  total_null_count = null_values.sum().sum()
  print("Total null/NaN values:", total_null_count)

def find_rows_and_cols_with_null(df):
  null_rows = df[df.isnull().any(axis=1)]
  print("Rows with null/NaN values:")
  print(null_rows)

  null_columns = df.columns[df.isnull().any()]
  print("Columns with null/NaN values:")
  print(null_columns)
  

# For each K in [1-30 (all numbers), 35-95 (in increments of 5), 100-1000 (in increments of 25)] (in total 80 different k values)
# Run k-means and Measure the values of all of 5 of the clustering validation metrics
def calc_scores(df, k):
  X = df.values
  kmeans = KMeans(n_clusters=k)
  labels = kmeans.fit_predict(X)

  print(f'labels {np.unique(labels)}')
  silhouette = silhouette_score(X, labels)
  print("Silhouette Coefficient:", silhouette)

  calinski_harabasz = calinski_harabasz_score(X, labels)
  print("Calinski-Harabasz Index:", calinski_harabasz)

  davies = davies_bouldin_score(X, labels)
  print("Davies-Bouldin score:", davies)

  elbow = kmeans.inertia_
  return elbow, davies, silhouette, calinski_harabasz

def kmeans_scores(df):
    X = df.drop('ground_truth', axis=1)
    k_values = list(range(2, 31)) + list(range(35, 96, 5)) + list(range(100, 1001, 25))
    k_scores = {}

    for k in k_values:
        elbow, davies, silhouette, calinski_harabasz =  calc_scores(X,k)
        k_scores['k'] = {'Elbow-Method': elbow, 'davies_bouldin_score': davies, 'silhouette_score': silhouette, 'calinski_harabasz_score': calinski_harabasz}

    return k_scores