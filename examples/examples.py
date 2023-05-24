
import os 
import sys 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from light_size_constrained_clustering import da

import numpy as np
import collections


def da_example():
    n_samples = 2000
    n_clusters = 3
    X = np.random.rand(n_samples, 2)
    model = da.DeterministicAnnealing(n_clusters, distribution=[0.1, 0.6, 0.3])
    model.fit(X)

    centers = model.cluster_centers_
    labels = model.labels_

    cluster_size = list(collections.Counter(labels).values())
    print("Cluster size: ", cluster_size)
    print("Cluster size count: ", [c / n_samples for c in cluster_size])

if __name__ == "__main__":
    da_example()
