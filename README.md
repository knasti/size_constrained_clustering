## Size Constrained Clustering Solver
[![Build Status](https://travis-ci.org/jingw2/size_constrained_clustering.svg?branch=master)](https://travis-ci.org/jingw2/size_constrained_clustering)
[![PyPI version](https://badge.fury.io/py/size-constrained-clustering.svg)](https://badge.fury.io/py/size-constrained-clustering)
![GitHub](https://img.shields.io/github/license/jingw2/size_constrained_clustering)
[![codecov](https://codecov.io/gh/jingw2/size_constrained_clustering/branch/master/graph/badge.svg)](https://codecov.io/gh/jingw2/size_constrained_clustering)
![PyPI - Downloads](https://img.shields.io/pypi/dm/size-constrained-clustering)
![Codecov](https://img.shields.io/codecov/c/github/jingw2/size_constrained_clustering)


Implementation of Deterministic Annealing Size Constrained Clustering. 
Size constrained clustering can be treated as an optimization problem. Details could be found in a set of reference paper.

This is a fork of https://github.com/jingw2/size_constrained_clustering that solves installation issues. And mantains only the Determinstic Annealing clustering.

### Installation
Requirement Python >= 3.6, Numpy >= 1.13, Cython >= 0.29
* install from PyPI
```shell
pip install light-size-constrained-clustering
```

### Methods
* Deterministic Annealling Algorithm: Input target cluster distribution, return correspondent clusters

### Usage:

Deterministic Annealing
```python
n_samples = 2000
n_clusters = 3
X = np.random.rand(n_samples, 2)
# distribution is the distribution of cluster sizes
model = da.DeterministicAnnealing(n_clusters, distribution=[0.1, 0.6, 0.3])
model.fit(X)
centers = model.cluster_centers_
labels = model.labels_
```
![alt text](https://github.com/jingw2/size_constrained_clustering/blob/master/pic/da.png)

Cluster size: 1200, 600 and 200 in the figure above, corresponding to distribution [0.6, 0.3, 0.1]


## Copyright
Copyright (c) 2023 Jing Wang & Albert Pla. Released under the MIT License. 

Third-party copyright in this distribution is noted where applicable.

### Reference
* [Clustering with Capacity and Size Constraints: A Deterministic
Approach](http://web.eecs.umich.edu/~mayankb/docs/ClusterCap.pdf)
* [Deterministic Annealing, Clustering and Optimization](https://thesis.library.caltech.edu/2858/1/Rose_k_1991.pdf)
* [Deterministic Annealing, Constrained Clustering, and Opthiieation](https://authors.library.caltech.edu/78353/1/00170767.pdf)
* [Shrinkage Clustering](https://www.researchgate.net/publication/322668506_Shrinkage_Clustering_A_fast_and_size-constrained_clustering_algorithm_for_biomedical_applications)
* [Clustering with size constraints](https://www.researchgate.net/publication/268292668_Clustering_with_Size_Constraints)
* [Data Clustering with Cluster Size Constraints Using a Modified k-means Algorithm](https://core.ac.uk/download/pdf/61217069.pdf)
* [KMeans Constrained Clustering Inspired by Minimum Cost Flow Problem](https://github.com/joshlk/k-means-constrained)
* [Same Size Kmeans Heuristics Methods](https://elki-project.github.io/tutorial/same-size_k_means)
* [Google's Operations Research tools's
`SimpleMinCostFlow`](https://developers.google.com/optimization/flow/mincostflow)
* [Cluster KMeans Constrained](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2000-65.pdf)
