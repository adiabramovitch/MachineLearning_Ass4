from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering

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

def get_bic_score(X, labels):
  # n is number of observations
  n, n_features = X.shape
  k = max(labels) + 1

  # Find each cluster center
  centroids = np.empty(shape=(k, X.shape[1]))
  for cluster in range(k):
    centroids[cluster] = np.mean(X[labels == cluster], axis=0)

  # get residual sum of squares for the bic equation
  rss = np.sum(np.linalg.norm(X - centroids[labels], axis=1) ** 2)
  bic = rss + np.log(n) * (k * n_features + k - 1)
  return bic

# For each K in [1-30 (all numbers), 35-95 (in increments of 5), 100-1000 (in increments of 25)] (in total 80 different k values)
# Run k-means and Measure the values of all of 5 of the clustering validation metrics
def calc_scores(X, model):
  elbow = 0
  labels = model.fit_predict(X)
  n_clusters = len(np.unique(labels))

  if n_clusters > 1 and n_clusters < X.shape[0]:
    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies = davies_bouldin_score(X, labels)
  else:
    silhouette = np.nan
    calinski_harabasz = np.nan
    davies = np.nan

  if isinstance(model, KMeans):
    elbow = model.inertia_

  bic = get_bic_score(X, labels)

  return elbow, davies, silhouette, calinski_harabasz, bic

def calc_scores_for_all(X):
    kmeans_scores, dbscan_scores, optics_scores, agg_scores  = {}, {}, {}, {}

    k_values = list(range(2, 31)) + list(range(35, 96, 5)) + list(range(100, 1001, 25))
    for k in k_values:
      kmeans = KMeans(n_clusters=k, n_init='auto')
      elbow, davies, silhouette, calinski_harabasz, bic = calc_scores(X, kmeans)
      kmeans_scores[k] = {'Elbow-Method': elbow, 'davies_bouldin_score': davies, 'silhouette_score': silhouette,
                          'calinski_harabasz_score': calinski_harabasz, 'bic_score': bic}

    for epsilon in np.arange(0.1, 2.1, 0.1):
      dbscan = DBSCAN(eps=epsilon)
      elbow, davies, silhouette, calinski_harabasz, bic = calc_scores(X, dbscan)
      dbscan_scores[epsilon] = {'Elbow-Method': elbow, 'davies_bouldin_score': davies, 'silhouette_score': silhouette,
                                'calinski_harabasz_score': calinski_harabasz, 'bic_score': bic}

    min_samples_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
    for min_sample in min_samples_values:
      optics = OPTICS(min_samples=min_sample)
      elbow, davies, silhouette, calinski_harabasz, bic = calc_scores(X, optics)
      optics_scores[min_sample] = {'Elbow-Method': elbow, 'davies_bouldin_score': davies, 'silhouette_score': silhouette,
                                   'calinski_harabasz_score': calinski_harabasz, 'bic_score': bic}

    n_clusters_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
    for n in n_clusters_values:
      agg = AgglomerativeClustering(n_clusters=n)
      elbow, davies, silhouette, calinski_harabasz, bic = calc_scores(X, agg)
      agg_scores[n] = {'Elbow-Method': elbow, 'davies_bouldin_score': davies, 'silhouette_score': silhouette,
                       'calinski_harabasz_score': calinski_harabasz, 'bic_score': bic}

    return kmeans_scores, dbscan_scores, optics_scores, agg_scores
