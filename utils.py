from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering, MiniBatchKMeans
from tqdm import tqdm
import wandb
import os

WANDB_PROJ_NAME = 'ML2023'
os.environ['WANDB_API_KEY'] = "49e20ebf47e19b7061f97e1e223790d896a6b31a"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_PROJECT"] = WANDB_PROJ_NAME


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


def plot_elbow_method(x_label, y_label, x_values, y_values, algo, dataset):
    # Plot SSE curve
    plt_name = f'{dataset}-{algo}:({y_label}) Plot'
    wandb.init(project=WANDB_PROJ_NAME, name=plt_name, config={"Dataset": dataset, "Metric": y_label})
    plt.plot(x_values, y_values, 'bo-')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    fig = plt.gcf()
    plt.show()
    wandb.log({f'Elbow Method': fig})
    fig.savefig(f'{plt_name}')
    wandb.finish()


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
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto', max_iter=1000)
        labels = kmeans.fit_predict(X)
        sse = kmeans.inertia_

    bic = get_bic_score(X, labels)

    return sse, davies, silhouette, calinski_harabasz, bic, n_clusters


def log_rows(algo, dataset, hyper_param, hyper_param_value, n_clusters, silhouette, davies, calinski_harabasz, bic,
             sse):
    metrics_dict = {'SSE-Elbow': sse, 'VRC': calinski_harabasz, 'DB': davies, 'Silhouette': silhouette, 'BIC': bic}
    for metric_name, metric_val in metrics_dict.items():
        run = wandb.init(project=WANDB_PROJ_NAME, config={"Algorithm": algo, "Dataset": dataset})
        run.log({'Hyper-parameter name': hyper_param, 'Hyper-parameter value': hyper_param_value,
                 'Metric name': metric_name, 'Metric value': metric_val, 'Num clusters': n_clusters})
        run.finish()
    # return [{'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param,
    #          'Hyper-parameter value': hyper_param_value,
    #          'Metric name': 'Silhouette', 'Metric value': silhouette, 'Num clusters': n_clusters},
    #         {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param,
    #          'Hyper-parameter value': hyper_param_value,
    #          'Metric name': 'DB', 'Metric value': davies, 'Num clusters': n_clusters},
    #         {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param,
    #          'Hyper-parameter value': hyper_param_value,
    #          'Metric name': 'CH', 'Metric value': calinski_harabasz, 'Num clusters': n_clusters},
    #         {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param,
    #          'Hyper-parameter value': hyper_param_value,
    #          'Metric name': 'BIC', 'Metric value': bic, 'Num clusters': n_clusters},
    #         {'Algorithm': algo, 'Dataset': dataset, 'Hyper-parameter name': hyper_param,
    #          'Hyper-parameter value': hyper_param_value,
    #          'Metric name': 'SSE-Elbow', 'Metric value': sse, 'Num clusters': n_clusters}]


def calc_scores_for_all(X, dataset):
    wandb.login()
    sse_arr = []
    davis_arr = []
    silhouette_arr = []
    vrc_arr = []
    bic_arr = []

    print("Working on KMeans")
    # For each K in [1-30 (all numbers), 35-95 (increments of 5), 100-1000 (increments of 25)] (80 different k values)
    # Run k-means and Measure the values of all of 5 of the clustering validation metrics
    k_values = list(range(1, 31)) + list(range(35, 96, 5)) + list(range(100, 1001, 25))
    for i in tqdm(range(len(k_values))):
        kmeans = KMeans(n_clusters=k_values[i], n_init='auto')
        sse, davies, silhouette, calinski_harabasz, bic, _ = calc_scores(X, kmeans)
        sse_arr.append(sse)
        davis_arr.append(davies)
        silhouette_arr.append(silhouette)
        vrc_arr.append(calinski_harabasz)
        bic_arr.append(bic)
        log_rows('K-Means', dataset, 'n_clusters', k_values[i], k_values[i], silhouette, davies, calinski_harabasz, bic, sse)
    plot_elbow_method('n_clusters (k)', 'SSE-Elbow', k_values, sse_arr, dataset)
    plot_elbow_method('n_clusters (k)', 'VRC', k_values, vrc_arr, dataset)
    plot_elbow_method('n_clusters (k)', 'DB', k_values, davis_arr, dataset)
    plot_elbow_method('n_clusters (k)', 'Silhouette', k_values, silhouette_arr, dataset)
    plot_elbow_method('n_clusters (k)', 'BIC', k_values, bic_arr, dataset)

    sse_arr.clear()
    davis_arr.clear()
    silhouette_arr.clear()
    vrc_arr.clear()
    bic_arr.clear()
    print("Working on DBSCAN")
    eps_values = np.arange(0.1, 2.1, 0.1)
    for i in tqdm(range(len(eps_values))):
        dbscan = DBSCAN(eps=eps_values[i], n_jobs=-1)
        sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, dbscan)
        sse_arr.append(sse)
        log_rows('dbscan', dataset, 'eps', eps_values[i], k, silhouette, davies, calinski_harabasz, bic, sse)
    plot_elbow_method('Epsilons', 'SSE-Elbow', eps_values, sse_arr, dataset)

    print("Working on OPTICS")
    sse_arr.clear()
    min_samples_values = range(2, 100, 5)
    for i in tqdm(range(len(min_samples_values))):
        optics = OPTICS(min_samples=min_samples_values[i], n_jobs=-1)
        sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, optics)
        sse_arr.append(sse)
        log_rows('OPTICS', dataset, 'min_samples', min_samples_values[i], k, silhouette, davies, calinski_harabasz, bic, sse)
    plot_elbow_method('Min Samples', 'SSE-Elbow', min_samples_values, sse_arr, dataset)

    print("Working on AgglomerativeClustering")
    sse_arr.clear()
    n_clusters_values = range(2, 100, 5)
    for i in tqdm(range(len(n_clusters_values))):
        agg = AgglomerativeClustering(n_clusters=n_clusters_values[i])
        sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, agg)
        sse_arr.append(sse)
        log_rows('Agglomerative', dataset, 'n_clusters', n_clusters_values[i], k, silhouette, davies, calinski_harabasz, bic, sse)
    plot_elbow_method('Clusters', 'SSE-Elbow', n_clusters_values, sse_arr, dataset)

# def calc_kmeans(X, dataset):
#     table = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
#                                   'Metric name', 'Metric value', 'Num clusters'])
#     sse_arr = []

#     print("Working on KMeans")
#     # For each K in [1-30 (all numbers), 35-95 (increments of 5), 100-1000 (increments of 25)] (80 different k values)
#     # Run k-means and Measure the values of all of 5 of the clustering validation metrics
#     k_values = list(range(1, 31)) + list(range(35, 96, 5)) + list(range(100, 1001, 25))
#     for i in tqdm(range(len(k_values))):
#         kmeans = KMeans(n_clusters=k_values[i], n_init='auto')
#         sse, davies, silhouette, calinski_harabasz, bic, _ = calc_scores(X, kmeans)
#         sse_arr.append(sse)
#         rows = create_rows('K-Means', dataset, 'n_clusters', k_values[i], k_values[i], silhouette, davies,
#                            calinski_harabasz, bic, sse)
#         table = pd.concat([table, pd.DataFrame(rows)])
#     plot_elbow_method('Clusters', 'SSE', k_values, sse_arr)
#     return table

# def calc_dbscan(X, dataset):
#     table = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
#                                   'Metric name', 'Metric value', 'Num clusters'])
#     sse_arr = []
#     print("Working on DBSCAN")
#     sse_arr.clear()
#     eps_values = np.arange(0.1, 2.1, 0.1)
#     for i in tqdm(range(len(eps_values))):
#         dbscan = DBSCAN(eps=eps_values[i], n_jobs=-1)
#         sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, dbscan)
#         sse_arr.append(sse)
#         rows = create_rows('DBSCAN', dataset, 'eps', eps_values[i], k, silhouette, davies, calinski_harabasz, bic, sse)
#         table = pd.concat([table, pd.DataFrame(rows)])
#     plot_elbow_method('Epsilons', 'SSE', eps_values, sse_arr)
#     return table

# def calc_optics(X, dataset):
#     table = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
#                                   'Metric name', 'Metric value', 'Num clusters'])
#     sse_arr = []
#     print("Working on OPTICS")
#     sse_arr.clear()
#     min_samples_values = range(2, 100, 5)
#     for i in tqdm(range(len(min_samples_values))):
#         optics = OPTICS(min_samples=min_samples_values[i], n_jobs=-1)
#         sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, optics)
#         sse_arr.append(sse)
#         rows = create_rows('OPTICS', dataset, 'min_samples', min_samples_values[i], k, silhouette, davies,
#                            calinski_harabasz, bic, sse)
#         table = pd.concat([table, pd.DataFrame(rows)])
#     plot_elbow_method('Min Sampels', 'SSE', min_samples_values, sse_arr)
#     return table

# def calc_agglomerativeClustering(X, dataset):
#     table = pd.DataFrame(columns=['Algorithm', 'Dataset', 'Hyper-parameter name', 'Hyper-parameter value',
#                                   'Metric name', 'Metric value', 'Num clusters'])
#     sse_arr = []
#     print("Working on OPTICS")
#     sse_arr.clear()
#     min_samples_values = range(2, 100, 5)
#     for i in tqdm(range(len(min_samples_values))):
#         optics = OPTICS(min_samples=min_samples_values[i], n_jobs=-1)
#         sse, davies, silhouette, calinski_harabasz, bic, k = calc_scores(X, optics)
#         sse_arr.append(sse)
#         rows = create_rows('OPTICS', dataset, 'min_samples', min_samples_values[i], k, silhouette, davies,
#                            calinski_harabasz, bic, sse)
#         table = pd.concat([table, pd.DataFrame(rows)])
#     plot_elbow_method('Min Sampels', 'SSE', min_samples_values, sse_arr)
#     return table
