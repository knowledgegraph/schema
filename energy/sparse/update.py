# -*- coding: utf-8 -*-

import numpy

import theano
import theano.tensor as T

import logging


# Note: RMS[g]t = sqrt(E[g^2]t + epsilon)
def RMS(value, epsilon=1e-6):
    return T.sqrt(value + epsilon)

def sgd(param, rate, gradient, updates):
    # Update: x_t - eta * g_t
    delta_x_t = - rate * gradient

    updates[param] = param + delta_x_t

def momentum(param, rate, decay, gradient, updates, param_previous_update):
    # decay represents the momentum
    delta_x_t = (decay * param_previous_update) - (rate * gradient)

    param_previous_update_updated = delta_x_t
    updates[param_previous_update] = param_previous_update_updated

    updates[param] = param + delta_x_t

def adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients):
    # ssg = \sum_t=1^T  [grad_t]^2
    param_squared_gradients_updated = param_squared_gradients + (gradient ** 2)
    updates[param_squared_gradients] = param_squared_gradients_updated

    # Update: x_t - (eta / sqrt(ssg)) * g_t
    delta_x_t = - (rate / RMS(param_squared_gradients_updated, epsilon=epsilon)) * gradient
    updates[param] = param + delta_x_t

def adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates):
    # Accumulate Gradient:
    # E[g^2]t = rho * E[g^2]t-1 + (1 - rho) * g^2_t
    param_squared_gradients_updated = (decay * param_squared_gradients) + ((1.0 - decay) * (gradient ** 2)) # Eg2_t = rho Eg2_t-1 + (1-rho) g2_t
    updates[param_squared_gradients] = param_squared_gradients_updated # E[g^2]t

    # Compute Update (Hessian approximation):
    #   [delta_x]t = - (RMS[delta_x]t-1 / RMS[g]t) g_t
    # Learning rate specified as in:
    #   http://climin.readthedocs.org/en/latest/adadelta.html
    delta_x_t = - rate * (RMS(param_squared_updates, epsilon=epsilon) / RMS(param_squared_gradients_updated, epsilon=epsilon)) * gradient

    # Accumulate updates:
    # E[delta_x^2]t = rho * E[delta_x^2]t-1 + (1 - rho) * [delta_x^2]t
    param_squared_updates_updated = (decay * param_squared_updates) + ((1.0 - decay) * (delta_x_t ** 2))

    updates[param_squared_updates] = param_squared_updates_updated
    # Apply update:
    # x_t+1 = x_t + [delta_x]t,
    #   as in x_t+1 = x_t + [delta_x]t, with [delta_x]t = - eta g_t
    updates[param] = param + delta_x_t

def rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients):
    # Accumulate Gradient:
    # E[g^2]t = rho * E[g^2]t-1 + (1 - rho) * g^2_t
    param_squared_gradients_updated = (decay * param_squared_gradients) + ((1.0 - decay) * (gradient ** 2)) # Eg2_t = rho Eg2_t-1 + (1-rho) g2_t
    updates[param_squared_gradients] = param_squared_gradients_updated # E[g^2]t

    # Compute Update:
    # [delta_x]t = - (eta / E[g^2]t) g_
    delta_x_t = - (rate / RMS(param_squared_gradients_updated, epsilon=epsilon)) * gradient

    # maxLearningRate ~as in https://github.com/w-cheng/optimx/blob/master/rmsprop.lua
    if (max_learning_rate is not None):
        max_rates = numpy.full(param.get_value().shape, max_learning_rate, dtype=theano.config.floatX)

        delta_x_t = T.minimum(delta_x_t, max_rates)
        # min_learning_rate mirrors max_learning_rate
        delta_x_t = T.maximum(delta_x_t, - max_rates)

    updates[param] = param + delta_x_t
