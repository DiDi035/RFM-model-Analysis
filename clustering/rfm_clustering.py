# AHP model
# To handle datetime
from datetime import datetime
from math import isnan
# Hopkins Statistics
from random import sample

import ahpy
# For Visualisation
import matplotlib.pyplot as plt
import numpy as np
# Importing Libraries
import pandas as pd
from numpy.random import uniform
# To perform KMeans clustering
from sklearn.cluster import KMeans
# For Silhouette Analysis
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, silhouette_score)
from sklearn.neighbors import NearestNeighbors
# To Scale our data
from sklearn.preprocessing import StandardScaler

import AHP
import random_forest as tree_classifier

BASE_DATE = datetime(1990, 1, 1)
END_DATE = datetime(2017, 12, 31)
CONVERTED_END_DATE = (END_DATE - BASE_DATE).days
TRANSACTION_DATA_PATH = \
    './data_handling/transaction_data/split/training/datasource_5.csv'
DEMOGRAPHIC_DATA_PATH = \
    './data_handling/customer_data/cleanned/datasource_5.csv'


def K_mean_model(RFM_norm, num_clusters):
    model_clus = KMeans(n_clusters=num_clusters, max_iter=100)
    model_clus.fit(RFM_norm)
    RFM_km = pd.concat([RFM, pd.Series(model_clus.labels_)], axis=1)
    RFM_km.columns = ['customer_id', 'total_amount',
                      'num_trans', 'recency', 'cluster_id']
    return RFM_km


def normalized_data(raw_data):
    # customer_id: {id, x, t_x, T}
    id_to_rfm_map = {
        'customer_id': [],
        'total_amount': [],
        'num_trans': [],
        'last_transaction': []
    }

    for index, row in raw_data.iterrows():
        existing_customer_index = id_to_rfm_map['customer_id'].index(
            row['customer_id']) if \
            row['customer_id'] in id_to_rfm_map['customer_id'] else -1
        if existing_customer_index != -1:
            id_to_rfm_map['total_amount'][existing_customer_index] = \
                id_to_rfm_map['total_amount'][existing_customer_index] \
                + float(row['total_amount'])
            id_to_rfm_map['num_trans'][existing_customer_index] = \
                id_to_rfm_map['num_trans'][existing_customer_index] \
                + float(row['num_trans'])
            id_to_rfm_map['last_transaction'][existing_customer_index] = max(
                id_to_rfm_map['last_transaction'][existing_customer_index],
                float(row['last_transaction']))
        else:
            id_to_rfm_map['customer_id'].append(row['customer_id'])
            id_to_rfm_map['total_amount'].append(float(row['total_amount']))
            id_to_rfm_map['num_trans'].append(float(row['num_trans']))
            id_to_rfm_map['last_transaction'].append(
                float(row['last_transaction']))

    return id_to_rfm_map


def treat_outlier(df, col_name, draw_boxplot=True):
    '''
    - Draws a boxplot for the given column name
    - Get rid of all outliers
    '''

    if draw_boxplot:
        plt.boxplot(df[col_name])
        plt.title(col_name)
        plt.show()
        plt.clf()

    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col_name] >= (Q1 - 1.5*IQR)) &
            (df[col_name] <= (Q3 + 1.5*IQR))]

    return df


def hopkins(X):
    '''
    The Hopkins statistic, is a statistic which gives a value which indicates\
    the cluster tendency, in other words: how well the data can be clustered.

    Some usefull links to understand Hopkins Statistics:
    - https://en.wikipedia.org/wiki/Hopkins_statistic
    - https://www.datanovia.com/en/lessons/assessing-clustering-tendency/
    '''

    d = X.shape[1]
    # d = len(vars) # columns
    n = len(X)  # rows
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(
            X, axis=0), d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(
            X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H


def silhouette_analysis(X, max_clusters, min_clusters=2, draw_plot=True):
    sse_ = []
    for k in range(min_clusters, max_clusters):
        kmeans = KMeans(n_clusters=k).fit(X)
        sse_.append([k, silhouette_score(X, kmeans.labels_)])

    if draw_plot:
        plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])
        plt.show()
        plt.clf()

    return sse_


# reading Dataset and extract RFM values
df = pd.read_csv(TRANSACTION_DATA_PATH, sep=',', encoding='ISO-8859-1',
                 header=0, low_memory=False)
RFM = df[['customer_id', 'total_amount', 'num_trans', 'last_transaction']]
norm_data = normalized_data(RFM)
RFM = pd.DataFrame(norm_data)

for idx, day in enumerate(RFM['last_transaction']):
    RFM.at[idx, 'last_transaction'] = CONVERTED_END_DATE - day

# outlier treatment
RFM = treat_outlier(RFM, 'total_amount', draw_boxplot=False)
RFM = treat_outlier(RFM, 'last_transaction', draw_boxplot=False)
RFM = treat_outlier(RFM, 'num_trans', draw_boxplot=False)

# standardise all parameters
RFM_norm1 = RFM.drop(['customer_id'], axis=1)
standard_scaler = StandardScaler()
RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)
RFM_norm1 = pd.DataFrame(RFM_norm1)
RFM_norm1.columns = ['Frequency', 'Amount', 'Recency']

# Calculate Hopkins Statist
# NOTE: hopkins_score = ~0.8767
# => The dataset has a high tendency to cluster!
# print(hopkins(RFM_norm1))

# Silhouette Analysis
# NOTE: Based on the result, k = 5 seems to be the best k for the dataset
# sse = silhouette_analysis(RFM_norm1, 7, draw_plot=True)

# Kmeans
RFM_km = K_mean_model(RFM_norm1, 2)

# km_clusters_amount = pd.DataFrame(
#     RFM_km.groupby(["cluster_id"]).total_amount.mean())
# km_clusters_frequency = pd.DataFrame(
#     RFM_km.groupby(["cluster_id"]).num_trans.mean())
# km_clusters_recency = pd.DataFrame(
#     RFM_km.groupby(["cluster_id"]).recency.mean())

# df = pd.concat([pd.Series([0, 1, 2, 3, 4]), km_clusters_recency,
#                km_clusters_frequency, km_clusters_amount], axis=1)
# df.columns = [
#     "cluster_id",
#     "recency_mean",
#     "frequency_mean",
#     "monetary_mean"
# ]

# sns.barplot(x=df.cluster_id, y=df.recency_mean)
# plt.title('recency_mean')
# print(df)

rfm_weights = ahpy.Compare(
    name='RFM model',
    comparisons=AHP.RFM_COMPARISIONS_1,
    precision=3,
    random_index='saaty')

rfm_weights_arr = np.array([
    -float(rfm_weights.target_weights['recency']),
    float(rfm_weights.target_weights['frequency']),
    float(rfm_weights.target_weights['monetary'])
])

num_users_each_cluster = RFM_km.groupby(["cluster_id"])['cluster_id'].count()

RFM_sum = np.array([
    RFM_km.groupby(["cluster_id"]).recency.mean().to_numpy(),
    RFM_km.groupby(["cluster_id"]).num_trans.mean().to_numpy(),
    RFM_km.groupby(["cluster_id"]).total_amount.mean().to_numpy()
])

result = rfm_weights_arr*RFM_sum.T
clv = []

for idx, rfm in enumerate(result):
    clv.append(np.sum(rfm)/num_users_each_cluster[idx])

highest_clv_cluster_idx = clv.index(max(clv))
customer_data = pd.read_csv(DEMOGRAPHIC_DATA_PATH,
                            sep=',', encoding='ISO-8859-1',
                            header=0, low_memory=False)
# make the data be the same format with DB
customer_data['customer_id'] = customer_data['id']
customer_data.drop('id', inplace=True, axis=1)

RFM_km_with_labels = tree_classifier \
    .pre_processing(customer_data, RFM_km, highest_clv_cluster_idx)

y_pred, y_test = tree_classifier.call(RFM_km_with_labels)
customer_data['labels'] = y_pred
result = customer_data[['customer_id', 'labels']]
print(result.to_dict('records'))
