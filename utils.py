from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from tqdm import tqdm


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

from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering
from tqdm import tqdm

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

def plot_elbow_method(x_label, y_label, x_values, y_values):
    # Plot SSE curve
    plt.plot( x_values, y_values, 'bo-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'Elbow Method')
    plt.show()

def calc_scores(X, model):
    sse = np.nan
    labels = model.fit_predict(X)
    n_clusters = len(np.unique(labels))

    if 1 < n_clusters < X.shape[0]:
        silhouette = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
    else:
        silhouette = np.nan
        calinski_harabasz = np.nan
        davies = np.nan

    if isinstance(model, KMeans):
        sse = model.inertia_
    else:
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X)
        sse = kmeans.inertia_

    bic = get_bic_score(X, labels)

    return sse, davies, silhouette, calinski_harabasz, bic, n_clusters

def create_rows(algo, dataset, hyper_param, hyper_param_value, n_clusters, silhouette, davies, calinski_harabasz, bic, sse):
  return [{'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param, 'Hyper-parameter value': hyper_param_value,
       'Metric name': 'Silhouette', 'Metric value': silhouette, 'Num clusters': n_clusters},
                {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param, 'Hyper-parameter value': hyper_param_value,
       'Metric name': 'DB', 'Metric value': davies, 'Num clusters': n_clusters},
                {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param, 'Hyper-parameter value': hyper_param_value,
       'Metric name': 'CH', 'Metric value': calinski_harabasz, 'Num clusters': n_clusters},
                {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param, 'Hyper-parameter value': hyper_param_value,
       'Metric name': 'BIC', 'Metric value': bic, 'Num clusters': n_clusters},
                {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param, 'Hyper-parameter value': hyper_param_value,
       'Metric name': 'SSE-Elbow', 'Metric value': sse, 'Num clusters': n_clusters}]

def calc_scores_for_all(X, dataset):
    table = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
                           'Metric name', 'Metric value', 'Num clusters'])
    sse_arr = []

    print("Working on KMeans")
    # For each K in [1-30 (all numbers), 35-95 (increments of 5), 100-1000 (increments of 25)] (80 different k values)
    # Run k-means and Measure the values of all of 5 of the clustering validation metrics
    k_values = list(range(1, 31)) + list(range(35, 96, 5)) + list(range(100, 1001, 25))
    for i in tqdm(range(len(k_values))):
        kmeans = KMeans(n_clusters=k_values[i], n_init='auto')
        sse, davies, silhouette, calinski_harabasz, bic, _ = calc_scores(X, kmeans)
        sse_arr.append(sse)
        rows = create_rows('K-Means', dataset, 'n_clusters', k_values[i], k_values[i], silhouette, davies, calinski_harabasz, bic, sse)
        table = pd.concat([table, pd.DataFrame(rows)])
    plot_elbow_method('Clusters', 'SSE', k_values, sse_arr)
    print(table.shape)

    print("Working on DBSCAN")
    sse_arr.clear()
    eps_values = np.arange(0.1, 2.1, 0.1)
    for i in tqdm(range(len(eps_values))):
        dbscan = DBSCAN(eps=eps_values[i], n_jobs=-1)
        sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, dbscan)
        sse_arr.append(sse)
        rows = create_rows('DBSCAN', dataset, 'eps', eps_values[i], k, silhouette, davies, calinski_harabasz, bic, sse)
        table = pd.concat([table, pd.DataFrame(rows)])
    plot_elbow_method('Epsilons', 'SSE', eps_values, sse_arr)

    print("Working on OPTICS")
    sse_arr.clear()
    min_samples_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
    for i in tqdm(range(len(min_samples_values))):
        optics = OPTICS(min_samples=min_samples_values[i], n_jobs=-1)
        sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, optics)
        sse_arr.append(sse)
        rows = create_rows('OPTICS', dataset, 'min_samples', min_samples_values[i], k ,silhouette, davies, calinski_harabasz, bic, sse)
        table = pd.concat([table, pd.DataFrame(rows)])
    plot_elbow_method('Min Sampels', 'SSE', min_samples_values, sse_arr)

    print("Working on AgglomerativeClustering")
    sse_arr.clear()
    n_clusters_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 45, 50]
    for i in tqdm(range(len(n_clusters_values))):
        agg = AgglomerativeClustering(n_clusters=n_clusters_values[i])
        sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, agg)
        sse_arr.append(sse)
        rows = create_rows('AgglomerativeClustering', dataset, 'n_clusters', n_clusters_values[i], k, silhouette, davies, calinski_harabasz, bic, sse)
        table = pd.concat([table, pd.DataFrame(rows)])
    plot_elbow_method('Clusters', 'SSE', n_clusters_values, sse_arr)

    return table
