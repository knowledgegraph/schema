# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import tsne
import sklearn
import sklearn.manifold
import sklearn.datasets

import logging


class DimensionalityReductionMethod(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def apply(self, X): pass


class TSNE(DimensionalityReductionMethod):
    def __init__(self, n_components=2):
        self.n_components = n_components
    def apply(self, X):
        fit = tsne.bh_sne(X, d=self.n_components)
        return fit


class MDS(DimensionalityReductionMethod):
    def __init__(self, n_components=2):
        self.mds = sklearn.manifold.MDS(n_components=n_components)
    def apply(self, X):
        fit = self.mds.fit(X)
        return fit.embedding_


class ISOMAP(DimensionalityReductionMethod):
    def __init__(self, n_neighbors=5, n_components=2):
        self.isomap = sklearn.manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)

    def apply(self, X):
        fit = self.isomap.fit(X)
        return fit.embedding_


class LLE(DimensionalityReductionMethod):
    def __init__(self, n_neighbors=5, n_components=2):
        self.lle = sklearn.manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components)

    def apply(self, X):
        fit = self.lle.fit(X)
        return fit.embedding_
