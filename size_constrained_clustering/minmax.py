#!usr/bin/python 3.7
#-*-coding:utf-8-*-

'''
@file: same_size_kmeans.py, equal size clustering with heuristics
@Author: Jing Wang (jingw2@foxmail.com)
@Date: 06/18/2020
@paper: 
@github reference: https://github.com/joshlk/k-means-constrained
@Web: 
'''

import os 
import sys
path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
import base 
from k_means_constrained import KMeansConstrained
import numpy as np
import collections
from sklearn.metrics.pairwise import haversine_distances
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

class MinMaxKMeansMinCostFlow(base.Base):

    def __init__(self, n_clusters, size_min=None, size_max=None, 
            max_iters=1000, distance_func=cdist, random_state=42):
        '''
        Args:
            n_clusters (int): number of clusters 
            max_iters (int): maximum iterations
            distance_func (object): callable function with input (X, centers) / None, by default is l2-distance
            random_state (int): random state to initiate, by default it is 42
        '''
        super(MinMaxKMeansMinCostFlow, self).__init__(n_clusters, max_iters, distance_func)
        self.clf = None
        self.size_min = size_min
        self.size_max = size_max 
        assert size_min is not None and size_max is not None
        assert size_min >= 0 and size_max >= 0
        assert size_min <= size_max

    def fit(self, X):
        n_samples, n_features = X.shape
        assert self.size_max * self.n_clusters >= n_samples

        clf = KMeansConstrained(self.n_clusters, size_min=self.size_min, 
            size_max=self.size_max, distance_func=self.distance_func)
   
        clf.fit(X)

        self.clf = clf 
        self.cluster_centers_ = self.clf.cluster_centers_
        self.labels_ = self.clf.labels_
    
    def predict(self, X):
        return self.clf.predict(X)

