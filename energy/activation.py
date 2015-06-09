# -*- coding: utf-8 -*-

import theano.tensor as T


# Activation functions
def htanh(x):
    return -1. * (x < -1.) + x * (x < 1.) * (x >= -1.) + 1. * (x >= 1)

def hsigm(x):
    return x * (x < 1) * (x > 0) + 1. * (x >= 1)

def rect(x):
    return x * (x > 0)

def sigm(x):
    return T.nnet.sigmoid(x)

def tanh(x):
    return T.tanh(x)

def lin(x):
    return x
