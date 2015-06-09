#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

# Classes of methods
base_vers = ['TransE', 'ScalE', 'NTransE', 'NScalE'] + ['BiTransE', 'BiScalE', 'BiNTransE', 'BiNScalE']

xi_vers = ['XiTransE', 'XiScalE', 'XiNTransE', 'XiNScalE']
xiscaltrans_vers = ['XiScalTransE', 'XiNScalTransE', 'XiTransScalE', 'XiNTransScalE']

semixi_vers = ['XiScalTransSE', 'XiTransScalSE', 'XiN1ScalTransSE']

lc_vers = ['CeTransE', 'CrTransE', 'CerTransE']
scaltrans_vers = ['ScalTransE', 'NScalTransE', 'BiScalTransE', 'BiNScalTransE']
aff_vers = ['AffinE', 'NAffinE', 'BiAffinE', 'BiNAffinE']
xiaff_vers = ['XiAffinE', 'XiNAffinE']

methods = base_vers + xi_vers + xiscaltrans_vers + semixi_vers + scaltrans_vers

for method in methods:
    print('./learn_parameters.py --seed=123 --strategy=ADAGRAD --totepochs=5 --test_all=100 --lr=0.100000 --train=data/wn/WN-train.pkl --valid=data/wn/WN-valid.pkl --test=data/wn/WN-test.pkl --nbatches=10 --no_rescaling --filtered  --op=%s --sim=L1 --ndim=5 --nhid=2 --margin=1' % (method))
