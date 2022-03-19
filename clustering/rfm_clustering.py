#Importing Libraries
import pandas as pd
import numpy as np

# For Silhouette Analysis
from sklearn.metrics import silhouette_score

# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# To Scale our data
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

# To perform KMeans clustering
from sklearn.cluster import KMeans

# Hopkins Statistics
from random import sample
from numpy.random import uniform
from math import isnan
from sklearn.neighbors import NearestNeighbors

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# To handle datetime
from datetime import datetime

BASE_DATE = datetime(1990,1,1)
END_DATE = datetime(2017,12,31) # first date of first quarter
START_DATE = datetime(2011,1,1) # last date of last quarter
CONVERTED_END_DATE = (END_DATE - BASE_DATE).days
DATA_PATH = '../data_handling/normalized_data/fact_quarter_transactions.csv'

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
    df = df[(df[col_name] >= (Q1 - 1.5*IQR)) & (df[col_name] <= (Q3 + 1.5*IQR))]

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
    n = len(X) # rows
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])

    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0

    return H

def silhouette_analysis(X, max_clusters, min_clusters = 2, draw_plot=True):
    sse_ = []
    for k in range(min_clusters, max_clusters):
        kmeans = KMeans(n_clusters=k).fit(X)
        sse_.append([k, silhouette_score(X, kmeans.labels_)])

    if draw_plot:
        plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);
        plt.show()
        plt.clf()

    return sse_

# reading Dataset and extract RFM values
df = pd.read_csv(DATA_PATH, sep = ',',encoding = 'ISO-8859-1', header=0, low_memory=False)
RFM = df[['customer_id', 'total_amount', 'num_trans', 'last_transaction']]

for idx, day in enumerate(RFM['last_transaction']):
  RFM.at[idx, 'last_transaction'] = CONVERTED_END_DATE - day

# # outlier treatment
RFM = treat_outlier(RFM, 'total_amount', draw_boxplot=False)
RFM = treat_outlier(RFM, 'last_transaction', draw_boxplot=False)
RFM = treat_outlier(RFM, 'num_trans', draw_boxplot=False)

# # standardise all parameters
RFM_norm1 = RFM.drop(['customer_id'], axis=1)
standard_scaler = StandardScaler()
RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)
RFM_norm1 = pd.DataFrame(RFM_norm1)
RFM_norm1.columns = ['Frequency','Amount','Recency']

# # Calculate Hopkins Statist
# NOTE: hopkins_score = ~0.994 => The dataset has a high tendency to cluster!
# print(hopkins(RFM_norm1))

# # Silhouette Analysis
# # NOTE: Based on the result, k = 5 seems to be the best k for the dataset
# sse = silhouette_analysis(RFM_norm1, 7, draw_plot=True)

# # Kmeans with K=5
model_clus5 = KMeans(n_clusters = 5, max_iter=50)
model_clus5.fit(RFM_norm1)
RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
RFM_km.columns = ['customer_id', 'total_amount', 'num_trans', 'recency', 'cluster_id']

km_clusters_amount = pd.DataFrame(RFM_km.groupby(["cluster_id"]).total_amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_km.groupby(["cluster_id"]).num_trans.mean())
km_clusters_recency = pd.DataFrame(RFM_km.groupby(["cluster_id"]).recency.mean())

df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
df.columns = ["cluster_id", "amount_mean", "frequency_mean", "recency_mean"]

sns.barplot(x=df.cluster_id, y=df.frequency_mean)
plt.title('frequency_mean')
plt.show()