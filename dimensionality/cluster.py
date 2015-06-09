# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy

import tsne
import sklearn
import sklearn.cluster
import sklearn.datasets

import logging


class ClusteringMethod(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def apply(self, X): pass


class NoClustering(ClusteringMethod):
    def __init__(self, n_clusters=1):
        self.n_clusters = n_clusters
    def apply(self, X):
        return numpy.zeros(shape=(X.shape[0]))


class KMeans(ClusteringMethod):
    def __init__(self, n_clusters=8):
        self.kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters)
    def apply(self, X):
        self.kmeans.fit(X)
        return self.kmeans.predict(X)


class AffinityPropagation(ClusteringMethod):
    def __init__(self, damping=0.5):
        self.affinity_propagation = sklearn.cluster.AffinityPropagation(damping=damping)
    def apply(self, X):
        self.affinity_propagation.fit(X)
        return self.affinity_propagation.predict(X)


class MeanShift(ClusteringMethod):
    def __init__(self, bandwidth=None):
        self.mean_shift = sklearn.cluster.MeanShift(bandwidth=bandwidth)
    def apply(self, X):
        self.mean_shift.fit(X)
        return self.mean_shift.predict(X)


class SpectralClustering(ClusteringMethod):
    def __init__(self, n_clusters=8):
        self.spectral_clustering = sklearn.cluster.SpectralClustering(n_clusters=n_clusters)
    def apply(self, X):
        self.spectral_clustering.fit(X)
        return self.spectral_clustering.fit_predict(X)


class AgglomerativeClustering(ClusteringMethod):
    def __init__(self, n_clusters=2):
        self.agglomerative_clustering = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clusters)
    def apply(self, X):
        self.agglomerative_clustering.fit(X)
        return self.agglomerative_clustering.predict(X)


class DBSCAN(ClusteringMethod):
    def __init__(self, eps=0.5):
        self.dbscan = sklearn.cluster.DBSCAN(eps=eps)
    def apply(self, X):
        self.dbscan.fit(X)
        return self.dbscan.predict(X)


class GMM(ClusteringMethod):
    def __init__(self, n_components=1):
        self.gmm = sklearn.cluster.GMM(n_components=n_components)
    def apply(self, X):
        self.gmm.fit(X)
        return self.gmm.predict(X)
