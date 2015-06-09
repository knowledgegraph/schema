# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T


# Similarity functions
def L1(left, right, axis=1):
    #return - T.sum(T.sqrt(T.sqr(left - right)), axis) # Causes NaN propagation
    return - T.sum(T.abs_(left - right), axis)

def L2(left, right, axis=1):
    return - T.sqrt(T.sum(T.sqr(left - right), axis))

def L2_sqr(left, right, axis=1):
    return - T.sum(T.sqr(left - right), axis)

def dot(left, right, axis=1):
    return T.sum(left * right, axis)

def cosine(left, right, axis=1):
    num = T.sum(left * right, axis)
    den = T.sqrt(T.sum(T.sqr(left), axis)) * T.sqrt(T.sum(T.sqr(right), axis))
    return num / den

def cosine_sqr(left, right, axis=1):
    num = T.sqr(T.sum(left * right, axis))
    den = T.sum(T.sqr(left), axis) * T.sum(T.sqr(right), axis)
    return num / den
