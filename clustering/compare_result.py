from sklearn import cluster
import rfm_clustering
import numpy as np

sorted_RFM = rfm_clustering.RFM.sort_values(by=['num_trans'], ascending=False).to_numpy()
clv_arr = np.array(rfm_clustering.clv)
highest_clv_cluster = float(np.where(clv_arr == np.amax(clv_arr))[0][0])
data_from_highest_cluster = rfm_clustering.RFM_km[rfm_clustering.RFM_km.cluster_id == highest_clv_cluster]

count = 0
for customer in sorted_RFM:
    if data_from_highest_cluster[data_from_highest_cluster.customer_id == customer[0]].empty == False:
        count += 1

print('###### sorted_RFM ###########')
print(sorted_RFM)
print('###### data_from_highest_cluster ###########')
print(data_from_highest_cluster)
print(count)
print((count/len(data_from_highest_cluster))*100)  # ~ 94 - 95%
