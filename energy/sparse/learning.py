# -*- coding: utf-8 -*-

import collections
import inspect

import numpy

import theano
import theano.tensor as T
import theano.sparse as S
import theano.gradient as G

import theano.tensor.shared_randomstreams as RS

import energy.loss as loss

import regularization as R
import update as U

import logging

class Embeddings(object):
    """Class for the embeddings matrix."""

    def __init__(self, rng, N, D, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param N: number of entities, relations or both.
        :param D: dimension of the embeddings.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D

        ebound = numpy.sqrt(6. / D)

        E_values = rng.uniform(low=-ebound, high=ebound, size=(D, N))
        E_values = E_values / numpy.sqrt(numpy.sum(E_values ** 2, axis=0))
        E_values = numpy.asarray(E_values, dtype=theano.config.floatX)

        self.E = theano.shared(value=E_values, name='E' + tag)

        # Define a normalization function with respect to the L_2 norm of the embedding vectors.
        self.updates = collections.OrderedDict({self.E : self.E / T.sqrt(T.sum(self.E ** 2, axis=0))})
        self.normalize = theano.function([], [], updates=self.updates)


def parse_embeddings(embeddings):
    """
    Utilitary function to parse the embeddings parameter in a normalized way for the Structured Embedding [Bordes et al., AAAI 2011] and the Semantic
    Matching Energy [Bordes et al., AISTATS 2012] models.
    """
    if type(embeddings) == list:
        embedding, relationl, relationr = embeddings[0], embeddings[1], embeddings[2]
    else:
        embedding, relationl, relationr = embeddings, embeddings, embeddings
    return embedding, relationl, relationr


def TrainFn(fnsim, embeddings, leftop, rightop,
                loss=loss.hinge, loss_margin=1.0, op='', method='SGD',
                decay=0.999, epsilon=1e-6, max_learning_rate=None,

                weight_L1_embed_regularizer=None, weight_L2_embed_regularizer=None,
                weight_L1_param_regularizer=None, weight_L2_param_regularizer=None,
                weight_contractive_regularizer_left=None, weight_contractive_regularizer_right=None):
    """
    This function returns a theano function to perform a training iteration, contrasting couples of positive and negative triplets. members are given
    as sparse matrices. for one positive triplet there is one negative triplet.

    :param fnsim: similarity function (on theano variables).
    :param embeddings: an embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    """

    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr, inpl, inpo = S.csr_matrix('inpr'), S.csr_matrix('inpl'), S.csr_matrix('inpo')
    inpln, inprn, inpon = S.csr_matrix('inpln'), S.csr_matrix('inprn'), S.csr_matrix('inpon')

    # Learning rates for parameters and embeddings
    rate_params = T.scalar('rate_params')
    rate_embeddings = T.scalar('rate_embeddings')

    # E: D x N, inp: N x B -> <E, inp>: D x B -> <E, inp>.T: B x D

    # Positive triplet functions
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T

    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T

    # Negative triplet functions
    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T

    relln = S.dot(relationl.E, inpon).T
    relrn = S.dot(relationr.E, inpon).T

    # Similarity Function, applied to g_lhs and g_rhs
    lop, rop = leftop(lhs, rell), rightop(rhs, relr)
    lopn, ropn = leftop(lhsn, relln), rightop(rhsn, relrn)

    simi = fnsim(lop, rop)
    simin = fnsim(lopn, ropn)

    supported_loss_args = inspect.getargspec(loss)[0]
    loss_args = {} if 'margin' not in supported_loss_args else { 'margin':loss_margin }

    cost, out = loss(simi, simin, **loss_args)

    # <EXPERIMENTAL_CODE>
    # Should I also plug examples from corrupted triples ?
    if weight_contractive_regularizer_left is not None:
        cost = cost + (weight_contractive_regularizer_left * R.contractive_regularizer(lop, lhs))

    if weight_contractive_regularizer_right is not None:
        cost = cost + (weight_contractive_regularizer_right * R.contractive_regularizer(rop, rhs))

    for rel_param in set([relationl.E, relationr.E]):
        if weight_L1_param_regularizer is not None:
            cost = cost + (weight_L1_param_regularizer * R.L1_regularizer(rel_param))
        if weight_L2_param_regularizer is not None:
            cost = cost + (weight_L2_param_regularizer * R.L2_regularizer(rel_param))

    if weight_L1_embed_regularizer is not None:
        cost = cost + (weight_L1_embed_regularizer * R.L1_regularizer(embedding.E))
    if weight_L2_embed_regularizer is not None:
        cost = cost + (weight_L2_embed_regularizer * R.L2_regularizer(embedding.E))
    # </EXPERIMENTAL_CODE>

    params = leftop.params + rightop.params + (fnsim.params if hasattr(fnsim, 'params') else [])
    params = list(set(params))

    embeds = [embedding.E] + ([relationr.E, relationl.E] if (type(embeddings) == list) else [])
    embeds = list(set(embeds))

    # The function updates the implicit function arguments according to the updates.
    updates = collections.OrderedDict()


    if (method == 'SGD'):
        pass # do nothing

    elif (method == 'MOMENTUM'):
        param_previous_update_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the previous updates
            previous_update_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_previous_update = theano.shared(value=previous_update_value, name='su_' + param.name)

            param_previous_update_map[param] = param_previous_update

    elif (method == 'ADAGRAD'):
        param_squared_gradients_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

    elif (method == 'ADADELTA'):
        param_squared_gradients_map = collections.OrderedDict()
        param_squared_updates_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

            # Allocate the sums of squared updates
            squared_updates_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_updates = theano.shared(value=squared_updates_value, name='su_' + param.name)

            param_squared_updates_map[param] = param_squared_updates

    elif (method == 'RMSPROP'):
        param_squared_gradients_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

    else:
        raise ValueError('Unknown method: %s' % (method))


    # Parameter Gradients
    gradientsparams = T.grad(cost, params)

    # Embeddings gradients
    gradientsembeds = T.grad(cost, embeds)


    # Learning Rates
    rates_params = [rate_params for i in range(len(params))]

    # In TransE etc. the rate for predicates' embeddings (that do not get normalized) is rate_params, not rate_embeddings
    rates_embeddings = [rate_embeddings, rate_params, rate_params] if len(embeds) > 1 else [rate_embeddings] # [rate_embeddings for i in range(len(embeds))]

    for param, gradient, rate in zip(params + embeds, gradientsparams + gradientsembeds, rates_params + rates_embeddings):

        if (method == 'SGD'): # SGD
            U.sgd(param, rate, gradient, updates)

        elif (method == 'MOMENTUM'): # SGD+MOMENTUM
            param_previous_update = param_previous_update_map[param]
            U.momentum(param, rate, decay, gradient, updates, param_previous_update)

        elif (method == 'ADAGRAD'): # ADAGRAD
            param_squared_gradients = param_squared_gradients_map[param]
            U.adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients)

        elif (method == 'ADADELTA'): # ADADELTA
            param_squared_gradients = param_squared_gradients_map[param]
            param_squared_updates = param_squared_updates_map[param]
            U.adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates)

        elif (method == 'RMSPROP'): # RMSPROP
            param_squared_gradients = param_squared_gradients_map[param]
            U.rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients)

        else:
            raise ValueError('Unknown method: %s' % (method))

    """
    Theano function inputs.
    :input rate_embeddings: learning/decay rate for the embeddings.
    :input rate_params: learning/decay rate for the parameters.

    :input inpl: sparse csr matrix representing the indexes of the positive triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inpr: sparse csr matrix representing the indexes of the positive triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpo: sparse csr matrix representing the indexes of the positive triplet relation member, shape=(#examples,N [Embeddings]).
    :input inpln: sparse csr matrix representing the indexes of the negative triplet 'left' member, shape=(#examples,N [Embeddings]).
    :input inprn: sparse csr matrix representing the indexes of the negative triplet 'right' member, shape=(#examples,N [Embeddings]).
    :input inpon: sparse csr matrix representing the indexes of the negative triplet relation member, shape=(#examples,N [Embeddings]).

    Theano function output.
    :output mean(cost): average cost.
    :output mean(out): ratio of examples for which the margin is violated,
                       i.e. for which an update occurs.
    """
    return theano.function([rate_embeddings, rate_params, inpl, inpr, inpo, inpln, inprn, inpon],
                           [T.mean(cost), T.mean(out)], updates=updates, on_unused_input='ignore')


def TrainFn1Member(fnsim, embeddings, leftop, rightop, rel=True,
                    loss=loss.hinge, loss_margin=1.0, op=None, method='SGD',
                    decay=0.999, epsilon=1e-6, max_learning_rate=None,

                    weight_L1_embed_regularizer=None, weight_L2_embed_regularizer=None,
                    weight_L1_param_regularizer=None, weight_L2_param_regularizer=None,
                    weight_contractive_regularizer_left=None, weight_contractive_regularizer_right=None):

    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr, inpl, inpo = S.csr_matrix('inpr'), S.csr_matrix('inpl'), S.csr_matrix('inpo')
    inpln, inprn = S.csr_matrix('inpln'), S.csr_matrix('inprn')

    # Learning rates for parameters and embeddings
    rate_params = T.scalar('rate_params')
    rate_embeddings = T.scalar('rate_embeddings')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T

    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T

    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T

    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr))
    # Negative 'left' member
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr))
    # Negative 'right' member
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr))

    costl, outl = loss(simi, similn, margin=loss_margin)
    costr, outr = loss(simi, simirn, margin=loss_margin)

    cost, out = costl + costr, T.concatenate([outl, outr])

    # List of inputs of the function
    list_in = [rate_embeddings, rate_params, inpl, inpr, inpo, inpln, inprn]

    if rel:
        # If rel is True, we also consider a negative relation member
        inpon = S.csr_matrix()

        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T

        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))

        costo, outo = loss(simi, simion, margin=loss_margin)
        cost += costo
        out = T.concatenate([out, outo])
        list_in += [inpon]

    # <EXPERIMENTAL_CODE>
    # Should I also plug examples from corrupted triples ?
    if weight_contractive_regularizer_left is not None:
        cost = cost + (weight_contractive_regularizer_left * R.contractive_regularizer(lop, lhs))

    if weight_contractive_regularizer_right is not None:
        cost = cost + (weight_contractive_regularizer_right * R.contractive_regularizer(rop, rhs))

    for rel_param in set([relationl.E, relationr.E]):
        if weight_L1_param_regularizer is not None:
            cost = cost + (weight_L1_param_regularizer * R.L1_regularizer(rel_param))
        if weight_L2_param_regularizer is not None:
            cost = cost + (weight_L2_param_regularizer * R.L2_regularizer(rel_param))

    if weight_L1_embed_regularizer is not None:
        cost = cost + (weight_L1_embed_regularizer * R.L1_regularizer(embedding.E))
    if weight_L2_embed_regularizer is not None:
        cost = cost + (weight_L2_embed_regularizer * R.L2_regularizer(embedding.E))
    # </EXPERIMENTAL_CODE>

    params = leftop.params + rightop.params + (fnsim.params if hasattr(fnsim, 'params') else [])

    embeds = [embedding.E] + ([relationr.E, relationl.E] if (type(embeddings) == list) else [])

    # The function updates the implicit function arguments according to the updates.
    updates = collections.OrderedDict()

    if (method == 'SGD'):
        pass # do nothing

    elif (method == 'MOMENTUM'):
        param_previous_update_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the previous updates
            previous_update_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_previous_update = theano.shared(value=previous_update_value, name='su_' + param.name)

            param_previous_update_map[param] = param_previous_update

    elif (method == 'ADAGRAD'):
        param_squared_gradients_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

    elif (method == 'ADADELTA'):
        param_squared_gradients_map = collections.OrderedDict()
        param_squared_updates_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)
            param_squared_gradients_map[param] = param_squared_gradients

            # Allocate the sums of squared updates
            squared_updates_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_updates = theano.shared(value=squared_updates_value, name='su_' + param.name)

            param_squared_updates_map[param] = param_squared_updates

    elif (method == 'RMSPROP'):
        param_squared_gradients_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

    else:
        raise ValueError('Unknown method: %s' % (method))

    # Parameter Gradients
    gradientsparams = T.grad(cost, params)

    # Embeddings gradients
    gradientsembeds = T.grad(cost, embeds)


    # Learning Rates
    rates_params = [rate_params for i in range(len(params))]

    # In TransE etc. the rate for predicates' embeddings (that do not get normalized) is rate_params, not rate_embeddings
    rates_embeddings = [rate_embeddings, rate_params, rate_params] if len(embeds) > 1 else [rate_embeddings] # [rate_embeddings for i in range(len(embeds))]

    for param, gradient, rate in zip(params + embeds, gradientsparams + gradientsembeds, rates_params + rates_embeddings):

        if (method == 'SGD'): # SGD
            U.sgd(param, rate, gradient, updates)

        elif (method == 'MOMENTUM'): # SGD+MOMENTUM
            param_previous_update = param_previous_update_map[param]
            U.momentum(param, rate, decay, gradient, updates, param_previous_update)

        elif (method == 'ADAGRAD'): # ADAGRAD
            param_squared_gradients = param_squared_gradients_map[param]
            U.adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients)

        elif (method == 'ADADELTA'): # ADADELTA
            param_squared_gradients = param_squared_gradients_map[param]
            param_squared_updates = param_squared_updates_map[param]
            U.adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates)

        elif (method == 'RMSPROP'): # RMSPROP
            param_squared_gradients = param_squared_gradients_map[param]
            U.rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients)

        else:
            raise ValueError('Unknown method: %s' % (method))

    return theano.function(list_in, [T.mean(cost), T.mean(out)], updates=updates, on_unused_input='ignore')


def SimFn(fnsim, embeddings, leftop, rightop, op=''):
    """
    This function returns a Theano function to measure the similarity score for sparse matrices inputs.

    :param fnsim: similarity function (on Theano variables).
    :param embeddings: an Embeddings instance.
    :param leftop: class for the 'left' operator.
    :param rightop: class for the 'right' operator.
    """

    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs

    inpr, inpl, inpo = S.csr_matrix('inpr'), S.csr_matrix('inpl'), S.csr_matrix('inpo')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T

    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T

    lop, rop = leftop(lhs, rell), rightop(rhs, relr)

    simi = fnsim(lop, rop)

    """
    Theano function inputs.
    :input inpl: sparse csr matrix (representing the indexes of the 'left' entities), shape=(#examples, N [Embeddings]).
    :input inpr: sparse csr matrix (representing the indexes of the 'right' entities), shape=(#examples, N [Embeddings]).
    :input inpo: sparse csr matrix (representing the indexes of the relation member), shape=(#examples, N [Embeddings]).

    Theano function output
    :output simi: matrix of score values.
    """
    return theano.function([inpl, inpr, inpo], [simi], on_unused_input='ignore')


#
# SCHEMA-AWARE FUNCTIONS
#

class Prior(object):
    def __init__(self, rng, N=1, D=1, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param tag: name of the embeddings for parameter declaration.
        """
        self.N = N
        self.D = D

        ebound = numpy.sqrt(6. / D)

        P_values = rng.uniform(low=0.0, high=ebound, size=(N, D))
        P_values = P_values
        P_values = numpy.asarray(P_values, dtype=theano.config.floatX)

        self.P = theano.shared(value=P_values, name='P' + tag)

        # Define a clamping function
        self.updates = collections.OrderedDict({self.P : T.switch(self.P >= 0.0, self.P, 0.0)})
        self.clamp = theano.function([], [], updates=self.updates)


def TrainFn1Member_Schema(fnsim, embeddings, prior, leftop, rightop, rel=True,
                            loss=loss.hinge, loss_margin=1.0, op=None, method='SGD',
                            decay=0.999, epsilon=1e-6, max_learning_rate=None,
                            weight_L1_embed_regularizer=None, weight_L2_embed_regularizer=None,
                            weight_L1_param_regularizer=None, weight_L2_param_regularizer=None,
                            weight_contractive_regularizer_left=None, weight_contractive_regularizer_right=None):

    embedding, relationl, relationr = parse_embeddings(embeddings)

    # Inputs
    inpr, inpl, inpo = S.csr_matrix('inpr'), S.csr_matrix('inpl'), S.csr_matrix('inpo')
    inpln, inprn = S.csr_matrix('inpln'), S.csr_matrix('inprn')

    # Learning rates for parameters and embeddings
    rate_params = T.scalar('rate_params')
    rate_embeddings = T.scalar('rate_embeddings')

    g, gln, grn = T.matrix('g'), T.matrix('gln'), T.matrix('grn')

    # Graph
    lhs = S.dot(embedding.E, inpl).T
    rhs = S.dot(embedding.E, inpr).T

    rell = S.dot(relationl.E, inpo).T
    relr = S.dot(relationr.E, inpo).T

    lhsn = S.dot(embedding.E, inpln).T
    rhsn = S.dot(embedding.E, inprn).T

    # relation-dependent penalty weights
    pen_simi = g[0, :].T * S.dot(prior.P[:, 0], inpo).T + g[1, :].T * S.dot(prior.P[:, 1], inpo).T
    pen_similn = gln[0, :].T * S.dot(prior.P[:, 0], inpo).T + gln[1, :].T * S.dot(prior.P[:, 1], inpo).T
    pen_simirn = grn[0, :].T * S.dot(prior.P[:, 0], inpo).T + grn[1, :].T * S.dot(prior.P[:, 1], inpo).T

    # sim(left, right) = - sum(abs(left - right)) (Negative Energy)
    simi = fnsim(leftop(lhs, rell), rightop(rhs, relr)) - pen_simi

    # Negative 'left' member (Negative Energy)
    similn = fnsim(leftop(lhsn, rell), rightop(rhs, relr)) - pen_similn

    # Negative 'right' member (Negative Energy)
    simirn = fnsim(leftop(lhs, rell), rightop(rhsn, relr)) - pen_simirn

    # hinge(pos, neg) = max(0, neg - pos + 1)
    costl, outl = loss(simi, similn, margin=loss_margin)
    costr, outr = loss(simi, simirn, margin=loss_margin)

    cost, out = costl + costr, T.concatenate([outl, outr])

    # List of inputs of the function
    list_in = [rate_embeddings, rate_params, inpl, inpr, inpo, inpln, inprn, g, gln, grn]

    if rel:
        # If rel is True, we also consider a negative relation member
        inpon = S.csr_matrix()

        relln = S.dot(relationl.E, inpon).T
        relrn = S.dot(relationr.E, inpon).T

        simion = fnsim(leftop(lhs, relln), rightop(rhs, relrn))

        costo, outo = loss(simi, simion, margin=loss_margin)
        cost += costo
        out = T.concatenate([out, outo])
        list_in += [inpon]


    params = leftop.params + rightop.params + (fnsim.params if hasattr(fnsim, 'params') else []) + [prior.P]
    embeds = [embedding.E] + ([relationr.E, relationl.E] if (type(embeddings) == list) else [])


    # XXX: Post-Training:
    params = [prior.P]
    embeds = []


    # The function updates the implicit function arguments according to the updates.
    updates = collections.OrderedDict()

    if (method == 'SGD'):
        pass # do nothing

    elif (method == 'MOMENTUM'):
        param_previous_update_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the previous updates
            previous_update_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_previous_update = theano.shared(value=previous_update_value, name='su_' + param.name)

            param_previous_update_map[param] = param_previous_update

    elif (method == 'ADAGRAD'):
        param_squared_gradients_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

    elif (method == 'ADADELTA'):
        param_squared_gradients_map = collections.OrderedDict()
        param_squared_updates_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)
            param_squared_gradients_map[param] = param_squared_gradients

            # Allocate the sums of squared updates
            squared_updates_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_updates = theano.shared(value=squared_updates_value, name='su_' + param.name)

            param_squared_updates_map[param] = param_squared_updates

    elif (method == 'RMSPROP'):
        param_squared_gradients_map = collections.OrderedDict()

        for param in params + embeds:
            # Allocate the sums of squared gradients
            squared_gradients_value = numpy.zeros(param.get_value().shape, dtype=theano.config.floatX)
            param_squared_gradients = theano.shared(value=squared_gradients_value, name='sg_' + param.name)

            param_squared_gradients_map[param] = param_squared_gradients

    else:
        raise ValueError('Unknown method: %s' % (method))

    # Parameter Gradients
    gradientsparams = T.grad(cost, params)

    # Embeddings gradients
    gradientsembeds = T.grad(cost, embeds)


    # Learning Rates
    rates_params = [rate_params for i in range(len(params))]

    # XXX: Post-Training:
    rates_embeddings = []
    if len(embeds) > 0:
        rates_embeddings = [rate_embeddings, rate_params, rate_params] if len(embeds) > 1 else [rate_embeddings]


    for param, gradient, rate in zip(params + embeds, gradientsparams + gradientsembeds, rates_params + rates_embeddings):

        if (method == 'SGD'): # SGD
            U.sgd(param, rate, gradient, updates)

        elif (method == 'MOMENTUM'): # SGD+MOMENTUM
            param_previous_update = param_previous_update_map[param]
            U.momentum(param, rate, decay, gradient, updates, param_previous_update)

        elif (method == 'ADAGRAD'): # ADAGRAD
            param_squared_gradients = param_squared_gradients_map[param]
            U.adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients)

        elif (method == 'ADADELTA'): # ADADELTA
            param_squared_gradients = param_squared_gradients_map[param]
            param_squared_updates = param_squared_updates_map[param]
            U.adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates)

        elif (method == 'RMSPROP'): # RMSPROP
            param_squared_gradients = param_squared_gradients_map[param]
            U.rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients)

        else:
            raise ValueError('Unknown method: %s' % (method))

    return theano.function(list_in, [T.mean(cost), T.mean(out)], updates=updates, on_unused_input='ignore')
