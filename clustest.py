# setup
from light_size_constrained_clustering import da
import numpy as np

n_samples = 40 # number cells in spot
n_clusters = 4 # distinct number of cell types
distribution= [0.4,0.3,0.2,0.1] # distribution of each cell type (form deconv)
seed = 17

print(np.sum(distribution))
np.random.seed(seed)

X = np.random.rand(n_samples, 2)
# distribution is the distribution of cluster sizes
model = da.DeterministicAnnealing(n_clusters, distribution= distribution, random_state=seed)

model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
print("Labels:")
print(labels)
print("Elements in cluster 0: ", np.count_nonzero(labels == 0))
print("Elements in cluster 1: ", np.count_nonzero(labels == 1))
print("Elements in cluster 2: ", np.count_nonzero(labels == 2))
print("Elements in cluster 3: ", np.count_nonzero(labels == 3))
