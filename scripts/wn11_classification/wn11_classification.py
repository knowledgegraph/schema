#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

# Classes of methods

base_vers = ['TransE', 'ScalE', 'NTransE', 'NScalE', 'BiTransE', 'BiScalE', 'BiNTransE', 'BiNScalE']
scaltrans_vers = ['ScalTransE', 'NScalTransE', 'BiScalTransE', 'BiNScalTransE']
xi_vers = ['XiTransE', 'XiScalE', 'XiNTransE', 'XiNScalE']
xiscaltrans_vers = ['XiScalTransE', 'XiNScalTransE']
semixi_vers = ['XiScalTransSE', 'XiTransScalSE', 'XiN1ScalTransSE']

simple_method_set = base_vers + scaltrans_vers + xi_vers + xiscaltrans_vers + semixi_vers

sim_set = ['L1', 'L2', 'dot']

margin_set = [1, 2, 10]
ndim_set = [20, 50, 100, 200]
nhid_set = [20, 50, 100, 200]

ndim_small_set = [20, 50]
nhid_small_set = [20, 50]

epochs = 100
nbatches = 10
lr = 0.1
seed = 123

train_path = 'data/wn11/WN11-train.pkl'
valid_path = 'data/wn11/WN11-valid.pkl'
test_path = 'data/wn11/WN11-test.pkl'

# ADAGRAD
# def adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients):
c, method = 0, 'ADAGRAD'

# def adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients):
cmd_adagrad = ('./learn_parameters.py --seed=%d --strategy=%s --totepochs=%d --test_all=10 --lr=%f --name=wn11_classification/wn11_%s_%d '
                ' --train=%s --valid=%s --test=%s --nbatches=%d --no_rescaling --filtered --classification'
                ' --op=%s --sim=%s --ndim=%d --nhid=%d --margin=%d ' # varying params
                ' > logs/wn11_classification/wn11_classification.%s.%s.%d.%d.%d.%d.log 2>&1')


for op in simple_method_set:
    for sim in sim_set:
        for ndim in ndim_set:
            nhid = ndim
            for margin in margin_set:
                print(cmd_adagrad % (seed, method, epochs, lr, op, c, train_path, valid_path, test_path, nbatches, op, sim, ndim, nhid, margin, op, sim, ndim, nhid, margin, c))
                c += 1
