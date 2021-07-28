
import os 
import sys 
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from src import fcm, equal, da, minmax, shrinkage

from sklearn.datasets import make_blobs

import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import collections

def fcm_example():
    n_samples = 2000
    n_clusters = 4 
    centers = [(-5, -5), (0, 0), (5, 5), (7, 10)]

    X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                    centers=centers, shuffle=False, random_state=42)

    model = fcm.FCM(n_clusters)
    model.fit(X)
    centers = model.cluster_centers_
    labels = model.labels_


def equal_example():
    n_samples = 2000
    n_clusters = 3
    X = np.random.rand(n_samples, 2)
    # model = equal.SameSizeKMeansMinCostFlow(n_clusters)
    model = equal.SameSizeKMeansHeuristics(n_clusters)
    model.fit(X)

    centers = model.cluster_centers_
    labels = model.labels_

    print("Cluster size count: ", collections.Counter(labels))

def minmax_example():
    n_samples = 2000
    n_clusters = 3
    X = np.random.rand(n_samples, 2)
    model = minmax.MinMaxKMeansMinCostFlow(n_clusters, size_min=400, size_max=800)
    model.fit(X)

    centers = model.cluster_centers_
    labels = model.labels_

    print("Cluster size count: ", collections.Counter(labels))

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

def shrinkage_example():
    n_samples = 1000
    n_clusters = 4 
    centers = [(-5, -5), (0, 0), (5, 5), (7, 10)]

    X, _ = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                    centers=centers, shuffle=False, random_state=42)

    model = shrinkage.Shrinkage(n_clusters, size_min=100)
    model.fit(X)
    centers = model.cluster_centers_
    labels = model.labels_


if __name__ == "__main__":
    shrinkage_example()
