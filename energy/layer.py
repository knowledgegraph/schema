# -*- coding: utf-8 -*-

import numpy
import theano
import theano.tensor as T

import energy.activation as activation

# x, y: B x E

# Layers
class Layer(object):
    """Class for a layer with one input vector w/o biases."""

    def __init__(self, rng, act, n_inp, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: activation function.
        :param n_inp: input dimension.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = act
        self.n_inp = n_inp
        self.n_out = n_out

        wbound = numpy.sqrt(6. / (n_inp + n_out))
        W_values = numpy.asarray(rng.uniform(low=-wbound, high=wbound, size=(n_inp, n_out)), dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W' + tag)

        self.params = [self.W]

    # B x E
    def __call__(self, x):
        """Forward function."""

        # act(<B x E, E x O>) -> B x O
        return self.act(T.dot(x, self.W))

class LayerLinear(object):
    """Class for a layer with two inputs vectors with biases."""

    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: activation function.
        :param n_inpl: dimension of the 'left' input.
        :param n_inpr: dimension of the 'right' input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = act
        self.n_inpl = n_inpl
        self.n_inpr = n_inpr
        self.n_out = n_out

        self.layerl = Layer(rng, activation.lin, n_inpl, n_out, tag='left' + tag)
        self.layerr = Layer(rng, activation.lin, n_inpr, n_out, tag='right' + tag)
        b_values = numpy.zeros((n_out), dtype=theano.config.floatX)

        self.b = theano.shared(value=b_values, name='b' + tag)
        self.params = self.layerl.params + self.layerr.params + [self.b]

    # B x E, B x E
    def __call__(self, x, y):
        """Forward function."""
        # act(B x O + B x O + B x O) [broadcast] -> B x O
        return self.act(self.layerl(x) + self.layerr(y) + self.b)

class LayerBilinear(object):
    """
    Class for a layer with bilinear interaction (n-mode vector-tensor product)
    on two input vectors with a tensor of parameters.
    """

    def __init__(self, rng, act, n_inpl, n_inpr, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: activation function.
        :param n_inpl: dimension of the 'left' input.
        :param n_inpr: dimension of the 'right' input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = act
        self.n_inpl = n_inpl
        self.n_inpr = n_inpr
        self.n_out = n_out
        wbound = numpy.sqrt(9. / (n_inpl + n_inpr + n_out))

        W_values = rng.uniform(low=-wbound, high=wbound, size=(n_inpl, n_inpr, n_out))
        W_values = numpy.asarray(W_values, dtype=theano.config.floatX)
        self.W = theano.shared(value=W_values, name='W' + tag)

        b_values = numpy.zeros((n_out), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b' + tag)

        self.params = [self.W, self.b]

    # B x E, B x E
    def __call__(self, x, y):
        """Forward function."""
        xW = T.tensordot(x, self.W, axes=([1], [0]))
        xWy = ((y.reshape((y.shape[0], y.shape[1], 1))) * xW).sum(1)
        return self.act(xWy + self.b)

class LayerMat(object):
    """
    Class for a layer with two input vectors, the 'right' member being a flat
    representation of a matrix on which to perform the dot product with the
    'left' vector [Structured Embeddings model, Bordes et al. AAAI 2011].
    """

    def __init__(self, act, n_inp, n_out):
        """
        Constructor.

        :param act: activation function.
        :param n_inp: input dimension.
        :param n_out: output dimension.

        :note: there is no parameter declared in this layer, the parameters
               are the embeddings of the 'right' member, therefore their
               dimension have to fit with those declared here: n_inp * n_out.
        """
        self.act = act
        self.n_inp = n_inp
        self.n_out = n_out
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        # More details on the class and constructor comments.
        ry = y.reshape((y.shape[0], self.n_inp, self.n_out))
        rx = x.reshape((x.shape[0], x.shape[1], 1))
        return self.act((rx * ry).sum(1))

class LayerTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x + y

class LayerXiTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self, start, end):
        """Constructor."""
        self.start = start
        self.end = end
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        _y = y[:, self.start:self.end]
        return x + _y

class LayerNTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self):
        """Constructor."""
        self.layerTrans = LayerTrans()
        self.layerNorm = LayerNorm()
        self.params = self.layerTrans.params + self.layerNorm.params + []

    def __call__(self, x, y):
        """Forward function."""
        return self.layerNorm(self.layerTrans(x, y))

class LayerXiNTrans(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self, start, end):
        """Constructor."""
        self.start = start
        self.end = end
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    def __call__(self, x, y):
        """Forward function."""
        _y = y[:, self.start:self.end]
        return self.layerNorm(x + _y)

class LayerCTrans(object):
    def __init__(self, layerEntityCombination, layerRelationCombination):
        self.layerTrans = LayerTrans()
        self.layerEComb = layerEntityCombination
        self.layerRComb = layerRelationCombination
        self.params = self.layerTrans.params + self.layerEComb.params + self.layerRComb.params + []

    def __call__(self, x, y):
        """Forward function."""
        return self.layerTrans(self.layerEComb(x, None), self.layerRComb(y, None))

class LayerScal(object):
    """
    Class for a layer with two input vectors that performs the product of
    the 'left member' and 'right member' i.e. scaling x by y.
    """
    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x * y

class LayerXiScal(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self, start, end):
        """Constructor."""
        self.start = start
        self.end = end
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        _y = y[:, self.start:self.end]
        return x * _y

class LayerNScal(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self):
        """Constructor."""
        self.layerScal = LayerScal()
        self.layerNorm = LayerNorm()
        self.params = self.layerScal.params + self.layerNorm.params + []

    def __call__(self, x, y):
        """Forward function."""
        return self.layerNorm(self.layerScal(x, y))

class LayerXiNScal(object):
    """
    Class for a layer with two input vectors that performs the sum of
    the 'left member' and 'right member' i.e. translating x by y.
    """
    def __init__(self, start, end):
        """Constructor."""
        self.start = start
        self.end = end
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    def __call__(self, x, y):
        """Forward function."""
        _y = y[:, self.start:self.end]
        return self.layerNorm(x * _y)

class Unstructured(object):
    """
    Class for a layer with two input vectors that performs the linear operator
    of the 'left member'.

    :note: The 'right' member is the relation, therefore this class allows to
    define an unstructured layer (no effect of the relation) in the same
    framework.
    """
    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, y):
        """Forward function."""
        return x

class LayerCombination(object):
    """
    Class for a layer which provides a linear combinations of the parameter
    vectors (1-mode vector-matrix product) given the weights in the input vector.
    """
    def __init__(self, rng, act, n_inp, n_out, tag=''):
        """
        Constructor.

        :param rng: numpy.random module for number generation.
        :param act: activation function.
        :param n_inp: dimension of the input.
        :param n_out: output dimension.
        :param tag: name of the layer for parameter declaration.
        """
        self.act = act
        self.n_inp = n_inp
        self.n_out = n_out

        wbound = numpy.sqrt(9. / (n_inp + n_out))

        W_values = rng.uniform(low=-wbound, high=wbound, size=(n_out, n_inp))
        W_values = numpy.asarray(W_values, dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, name='W' + tag)

        self.params = [self.W]

    # B x E, B x E
    def __call__(self, x, y):
        """Forward function."""
        c = T.tensordot(x.T, self.W, axes=[0, 1])
        return self.act(c)

class LayerScalTrans(object):
    def __init__(self, ndim):
        """Constructor."""
        self.ndim = ndim
        self.params = []

    def __call__(self, x, st):
        """Forward function."""
        return (x * st[:, :self.ndim]) + st[:, -self.ndim:]

class LayerXiScalTrans(object):
    def __init__(self, ndim, start, end):
        """Constructor."""
        self.ndim = ndim
        self.start = start
        self.end = end
        self.params = []

    def __call__(self, x, st):
        """Forward function."""
        _st = st[:, self.start:self.end]
        return (x * _st[:, :self.ndim]) + _st[:, -self.ndim:]

class LayerXiN1ScalTrans(object):
    def __init__(self, ndim, start, end):
        """Constructor."""
        self.ndim = ndim
        self.start = start
        self.end = end
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    def __call__(self, x, st):
        """Forward function."""
        _st = st[:, self.start:self.end]
        return self.layerNorm(x * _st[:, :self.ndim]) + _st[:, -self.ndim:]
        #return (x * _st[:, :self.ndim]) + _st[:, -self.ndim:]

class LayerXiTransScal(object):
    def __init__(self, ndim, start, end):
        """Constructor."""
        self.ndim = ndim
        self.start = start
        self.end = end
        self.params = []

    def __call__(self, x, st):
        """Forward function."""
        _st = st[:, self.start:self.end]
        return (x + _st[:, :self.ndim]) * _st[:, -self.ndim:]

class LayerNScalTrans(object):
    def __init__(self, ndim):
        """Constructor."""
        self.ndim = ndim
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    def __call__(self, x, st):
        """Forward function."""
        return self.layerNorm((x * st[:, :self.ndim]) + st[:, -self.ndim:])

class LayerXiNScalTrans(object):
    def __init__(self, ndim, start, end):
        """Constructor."""
        self.ndim = ndim
        self.start = start
        self.end = end
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    def __call__(self, x, st):
        """Forward function."""
        _st = st[:, self.start:self.end]
        return self.layerNorm(x * _st[:, :self.ndim] + _st[:, -self.ndim:])

class LayerXiNTransScal(object):
    def __init__(self, ndim, start, end):
        """Constructor."""
        self.ndim = ndim
        self.start = start
        self.end = end
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    def __call__(self, x, st):
        """Forward function."""
        _st = st[:, self.start:self.end]
        return self.layerNorm(x + _st[:, :self.ndim] * _st[:, -self.ndim:])

class LayerAffin(object):
    def __init__(self, ndim, nhid):
        """Constructor."""
        self.ndim = ndim
        self.nhid = nhid
        self.params = []

    # B x E
    def __call__(self, x, aff):
        """Forward function."""
        _x = x.reshape((x.shape[0], x.shape[1], 1))
        _M = aff.reshape((aff.shape[0], self.ndim, self.nhid))
        faff = (_x * _M).sum(1)
        return faff

class LayerXiAffin(object):
    def __init__(self, ndim, nhid, start, end):
        """Constructor."""
        self.ndim = ndim
        self.nhid = nhid
        self.start = start
        self.end = end
        self.params = []

    # B x E
    def __call__(self, x, aff):
        """Forward function."""
        _aff = aff[:, self.start:self.end]
        _x = x.reshape((x.shape[0], x.shape[1], 1))
        _M = _aff.reshape((_aff.shape[0], self.ndim, self.nhid))
        faff = (_x * _M).sum(1)
        return faff

class LayerNAffin(object):
    def __init__(self, ndim, nhid):
        """Constructor."""
        self.ndim = ndim
        self.nhid = nhid
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    # B x E
    def __call__(self, x, aff):
        """Forward function."""
        _x = x.reshape((x.shape[0], x.shape[1], 1))
        _M = aff.reshape((aff.shape[0], self.ndim, self.nhid))
        faff = (_x * _M).sum(1)
        return self.layerNorm(faff)

class LayerXiNAffin(object):
    def __init__(self, ndim, nhid, start, end):
        """Constructor."""
        self.ndim = ndim
        self.nhid = nhid
        self.start = start
        self.end = end
        self.layerNorm = LayerNorm()
        self.params = self.layerNorm.params + []

    # B x E
    def __call__(self, x, aff):
        """Forward function."""
        _aff = aff[:, self.start:self.end]
        _x = x.reshape((x.shape[0], x.shape[1], 1))
        _M = _aff.reshape((_aff.shape[0], self.ndim, self.nhid))
        faff = (_x * _M).sum(1)
        return self.layerNorm(faff)

class LayerProjection(object):
    """
    Class for a layer with two input vectors that performs a projection
    of the 'left member' w.r.t. the 'right member'.
    """
    def __init__(self):
        """Constructor."""
        self.params = []

    def __call__(self, x, w):
        """Projection function."""
        projection = x - (T.sum(x * w, axis=1) * w.T).T
        return projection

class LayerNorm(object):
    """
    Class for a layer with one input vector that performs a normalization
    of the input (L2 norm).
    """
    def __init__(self):
        """Constructor."""
        self.params = []

    # B x D
    def __call__(self, x):
        """Normalization function."""
        norm = (x.T / T.sum(x.T ** 2, axis=0)).T
        return norm
