# -*- coding: utf-8 -*-

import theano
import theano.tensor as T


# Cost functions

def hinge(pos, neg, margin=1.0): # max(0, neg - pos + margin)
    out = neg - pos + margin
    return T.sum(out * (out > 0)), out > 0
