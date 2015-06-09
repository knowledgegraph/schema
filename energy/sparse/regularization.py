# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import theano.sparse as S
import theano.gradient as G


# Note: g(E) = \sum_{e \in E} [ ||e||^2_2 - 1 ]_+
def unitary_norm2_penalty(E, axis=1):
    value = T.sum(T.sqr(E), axis=axis) - 1.0   # ||e||^2_2 - 1
    hinge = value * (value > 0)           # [ ||e||^2_2 - 1 ]_+
    return T.sum(hinge)                   # sum_e [ ||e||^2_2 - 1 ]_+


# Note: g(E) = \sum_{r} [ ((w_r^T d_r)^2 / ||d_r||_2^2) - epsilon^2  ]_+
def orthogonal_penalty(W, D, epsilon=1e-6, axis=1):
    num = T.sqr(T.sum(W * D, axis=axis))                 # n = (d^T w)^2
    den = T.sum(T.sqr(W), axis=axis) * T.sum(T.sqr(D), axis=axis)  # d = ||w||_2^2 * ||d||_2^2
    cos = num / den                                      # c = n / d
    value = cos - (epsilon**2)                           # v = c - epsilon^2
    hinge = value * (value > 0)                          # h = [ v ]_+
    return T.sum(hinge)

# Note: r(W) = ||W||_1
def L1_regularizer(param):
    # Symbolic Theano variable that represents the L1 regularization term
    L1  = T.sum(T.abs_(param))
    return L1

# Note: r(W) = ||W||_2^2
def L2_regularizer(param):
    # Symbolic Theano variable that represents the squared L2 term
    L2_sqr = T.sum(param ** 2)
    return L2_sqr

def contractive_regularizer(op, examples):
    jacobian = G.jacobian(op.flatten(), examples)
    regularizer = T.sum(T.abs_(jacobian) ** 2)
    return regularizer
