# -*- coding: utf-8 -*-

import numpy
import energy.activation as activation
import energy.layer as layer


def op(op, ndim, nhid):

    # operators
    if op == 'Unstructured':
        leftop = layer.Unstructured()
        rightop = layer.Unstructured()
    elif op == 'SME_lin':
        leftop = layer.LayerLinear(numpy.random, activation.lin, ndim, ndim, nhid, 'left')
        rightop = layer.LayerLinear(numpy.random, activation.lin, ndim, ndim, nhid, 'right')
    elif op == 'SME_bil':
        leftop = layer.LayerBilinear(numpy.random, activation.lin, ndim, ndim, nhid, 'left')
        rightop = layer.LayerBilinear(numpy.random, activation.lin, ndim, ndim, nhid, 'right')
    elif op == 'SE':
        leftop = layer.LayerMat(activation.lin, ndim, nhid)
        rightop = layer.LayerMat(activation.lin, ndim, nhid)
    elif op == 'TransE':
        leftop = layer.LayerTrans()
        rightop = layer.Unstructured()
    elif op == 'BiTransE':
        leftop = layer.LayerTrans()
        rightop = layer.LayerTrans()
    elif op == 'XiTransE':
        leftop = layer.LayerXiTrans(start=0, end=ndim)
        rightop = layer.LayerXiTrans(start=ndim, end=ndim * 2)
    elif op == 'NTransE':
        leftop = layer.LayerNTrans()
        rightop = layer.Unstructured()
    elif op == 'XiNTransE':
        leftop = layer.LayerXiNTrans(start=0, end=ndim)
        rightop = layer.LayerXiNTrans(start=ndim, end=ndim * 2)
    elif op == 'BiNTransE':
        leftop = layer.LayerNTrans()
        rightop = layer.LayerNTrans()
    elif op == 'ScalE':
        leftop = layer.LayerScal()
        rightop = layer.Unstructured()
    elif op == 'XiScalE':
        leftop = layer.LayerXiScal(start=0, end=ndim)
        rightop = layer.LayerXiScal(start=ndim, end=ndim * 2)
    elif op == 'BiScalE':
        leftop = layer.LayerScal()
        rightop = layer.LayerScal()
    elif op == 'NScalE':
        leftop = layer.LayerNScal()
        rightop = layer.Unstructured()
    elif op == 'XiNScalE':
        leftop = layer.LayerXiNScal(start=0, end=ndim)
        rightop = layer.LayerXiNScal(start=ndim, end=ndim * 2)
    elif op == 'BiNScalE':
        leftop = layer.LayerNScal()
        rightop = layer.LayerNScal()
    elif op == 'ScalTransE':
        leftop = layer.LayerScalTrans(ndim)
        rightop = layer.Unstructured()
    elif op == 'BiScalTransE':
        leftop = layer.LayerScalTrans(ndim)
        rightop = layer.LayerScalTrans(ndim)
    elif op == 'XiScalTransE':
        leftop = layer.LayerXiScalTrans(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiScalTrans(ndim, start=ndim * 2, end=ndim * 4)
    elif op == 'XiScalTransSE':
        leftop = layer.LayerXiScalTrans(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiScal(start=(ndim * 2), end=(ndim * 3))
    elif op == 'XiN1ScalTransSE':
        leftop = layer.LayerXiN1ScalTrans(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiNScal(start=(ndim * 2), end=(ndim * 3))
    elif op == 'XiTransScalE':
        leftop = layer.LayerXiTransScal(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiTransScal(ndim, start=ndim * 2, end=ndim * 4)
    elif op == 'XiTransScalSE':
        leftop = layer.LayerXiTransScal(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiTrans(start=ndim * 2, end=ndim * 3)
    elif op == 'NScalTransE':
        leftop = layer.LayerNScalTrans(ndim)
        rightop = layer.Unstructured()
    elif op == 'BiNScalTransE':
        leftop = layer.LayerNScalTrans(ndim)
        rightop = layer.LayerNScalTrans(ndim)
    elif op == 'XiNScalTransE':
        leftop = layer.LayerXiNScalTrans(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiNScalTrans(ndim, start=ndim * 2, end=ndim * 4)
    elif op == 'XiNTransScalE':
        leftop = layer.LayerXiNTransScal(ndim, start=0, end=ndim * 2)
        rightop = layer.LayerXiNTransScal(ndim, start=ndim * 2, end=ndim * 4)
    elif op == 'AffinE':
        leftop = layer.LayerAffin(ndim, ndim)
        rightop = layer.Unstructured()
    elif op == 'NAffinE':
        leftop = layer.LayerNAffin(ndim, ndim)
        rightop = layer.Unstructured()
    elif op == 'BiAffinE':
        leftop = layer.LayerAffin(ndim, nhid)
        rightop = layer.LayerAffin(ndim, nhid)
    elif op == 'XiAffinE':
        leftop = layer.LayerXiAffin(ndim, nhid, start=0, end=ndim * nhid)
        rightop = layer.LayerXiAffin(ndim, nhid, start=ndim * nhid, end=(ndim * nhid) * 2)
    elif op == 'BiNAffinE':
        leftop = layer.LayerNAffin(ndim, nhid)
        rightop = layer.LayerNAffin(ndim, nhid)
    elif op == 'XiNAffinE':
        leftop = layer.LayerXiNAffin(ndim, nhid, start=0, end=ndim * nhid)
        rightop = layer.LayerXiNAffin(ndim, nhid, start=ndim * nhid, end=(ndim * nhid) * 2)
    elif op == 'CeTransE':
        cemb = layer.LayerCombination(numpy.random, activation.lin, ndim, nhid, 'cemb')
        crel = layer.Unstructured()
        leftop = layer.LayerCTrans(cemb, crel)
        rightop = cemb
    elif op == 'CrTransE':
        cemb = layer.Unstructured()
        crel = layer.LayerCombination(numpy.random, activation.lin, ndim, nhid, 'cemb')
        leftop = layer.LayerCTrans(cemb, crel)
        rightop = cemb
    elif op == 'CerTransE':
        cemb = layer.LayerCombination(numpy.random, activation.lin, ndim, nhid, 'cemb')
        crel = layer.LayerCombination(numpy.random, activation.lin, ndim, nhid, 'cemb')
        leftop = layer.LayerCTrans(cemb, crel)
        rightop = cemb
    else:
        raise ValueError('Unknown model: %s' % (op))

    return (leftop, rightop)
