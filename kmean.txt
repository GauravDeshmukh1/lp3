import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#read dataset 
dataset = pd.read_csv("Kmeans_dataset.csv")
print(dataset)

f1 = dataset['X'].values
f2 = dataset['Y'].values
x = np.array(list(zip(f1, f2)))
print(x)

# initial centroid points
centers = np.array([[0.1,0.6],[0.3,0.2]])
print(centers)

# Apply K-Means Clustering
from sklearn.cluster import KMeans 
model = KMeans(n_clusters=2, init=centers, n_init=1)
# n_clusters = number of clusters
# init = initial centroids
# n_init = number of init parameter

#train the algorithm
model.fit(x)

# print labels
print(model.labels_)


# population around cluster 2
print(np.count_nonzero(model.labels_ == 1))

# new centroids 
print(model.cluster_centers_)

# Cluster with initial centroids
C_x = np.array([0.1, 0.3])
C_y = np.array([0.6, 0.2]) 
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
# cluster with new centroids 
N_x = np.array([model.cluster_centers_[0][0], model.cluster_centers_[1][0]])
N_y = np.array([model.cluster_centers_[0][1],model.cluster_centers_[1][1]])
plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(N_x, N_y, marker='*', s=200, c='g')