#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import os, sys, types, getopt, operator, logging

import numpy as np
import cPickle as pickle
import rdflib

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def toexp(num):
    if num < 0.2:
        pow = int(np.log10(float(str(num))))
        ret = str('10^{%s}' % pow)

        if ret == '10^{-1}':
            ret = '0.1'

    else:
        ret = str(num)
    print('::: %s -> %s' % (float(str(num)), ret))
    return ret


def process(path):
    f = open(path, 'rb')
    document = pickle.load(f)
    f.close()

    if 'state' not in document.keys():
        return None

    state = document['state']
    _state = {k: (state[k].__name__ if isinstance(state[k], types.FunctionType) else str(state[k])) for k in state.keys()}

    logging.info('State: %s' % str(_state))

    # Method
    method = state['method']
    method_name = method.upper()

    # Parameters
    lr, decay, ndim, aeps, max_lr = state['lremb'], state['decay'], state['ndim'], state['epsilon'], state['max_lr']

    name = ''

    if method_name == 'ADAGRAD':
        method_name = 'AdaGrad'

    if method_name == 'ADADELTA':
        if (1.0 - float(decay)) < 0.2:
            _decay = toexp(1.0 - float(decay))
        else:
            _decay = str(1.0 - float(decay))
        #print('XXX %s ' % _decay)
        _decay2 = toexp(1.0 - float(decay))
        name = 'AdaDelta' + ' $(1 - \\rho) = ' + _decay2 + ', \\epsilon=' + toexp(float(aeps)) + '$'
    elif method_name == 'RMSPROP':
        name = 'RMSprop' + ' $(1 - \\rho) = ' + str(1.0 - float(decay)) + ', \\eta=' + toexp(float(lr)) + ', \\omega=' + toexp(float(max_lr)) + '$'
    elif method_name == 'MOMENTUM':
        name = 'Momentum' + ' $\\eta = ' + toexp(float(lr)) +  ', (1 - \\rho) = ' + str(1.0 - float(decay)) + '$'
    else:
        name = method_name + ' $\\eta = ' + toexp(float(lr)) + '$'

    logging.info('Name: %s' % (name))

    if 'average_costs_per_epoch' in document.keys():
        average_costs_per_epoch = document['average_costs_per_epoch']

        # [ (mean, std), (mean, std), (mean, std), .. ]
        costs_per_epoch = [(np.mean(epoch_costs), np.std(epoch_costs)) for epoch_costs in average_costs_per_epoch]
        logging.debug(name + ' = ' + str(['{:.3f}'.format(mean_cost) for (mean_cost, _) in costs_per_epoch]).replace('\'', ''))
        mean_costs = [mean_cost for (mean_cost, _) in costs_per_epoch]

    ret = (name, mean_costs)

    return ret


def main(argv):

    for arg in argv:
        if (arg == '-h' or arg == '--help'):
            logging.info('Sample usage: LOSS_THR=200 SAVE_FILE=aifb_adadelta_rescaled_200.png %s ~/models/*.pkl' % (sys.argv[0]))
            return

    loss_threshold = None
    if ('LOSS_THR' in os.environ.keys()):
        loss_threshold = float(os.environ['LOSS_THR'])

    font_scale = None
    if ('FONT_SCALE' in os.environ.keys()):
        font_scale = float(os.environ['FONT_SCALE'])

    epochs = None
    if ('EPOCHS' in os.environ.keys()):
        epochs = int(os.environ['EPOCHS'])

    title_str = None
    if ('TITLE' in os.environ.keys()):
        plt.rc('text', usetex=True)
        title_str = os.environ['TITLE']

    name_lines = []

    is_show = False

    for arg in argv:
        logging.info('Processing %s ..' % (arg))

        if arg == '-show':
            is_show = True
        else:
            res = process(arg)
            logging.info(res)
            if res is not None:
                name_lines += [res]

    name_lines_dict = {}
    for name_line in name_lines:
        name, line = name_line
        if name not in name_lines_dict:
            name_lines_dict[name] = []
        name_lines_dict[name] += [line[:epochs]]

    new_name_lines_dict = {}
    scores = {}

    for name in name_lines_dict.keys():
        lines = name_lines_dict[name]
        mean = np.mean([line[-1] for line in lines])
        scores[name] = mean
        if (loss_threshold is None) or (mean < loss_threshold):
            new_name_lines_dict[name] = lines
            print('%s : %d' % (name, len(lines)))

    name_lines_dict = new_name_lines_dict

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
    print(sorted_scores)

    if ('TOP_K' in os.environ.keys()):
        top_k = int(os.environ['TOP_K'])
        sorted_scores = sorted_scores[:top_k]

    if ('BEST_K' in os.environ.keys()):
        best_k = int(os.environ['BEST_K'])
        methods = [name.split(' ')[0] for (name, score) in sorted_scores]
        _sorted_scores = []
        for method in set(methods):
            c = 0
            for (name, score) in sorted_scores:
                if name.split(' ')[0] == method and c < best_k:
                    print(': %s' % (str((name, score))))
                    _sorted_scores += [(name, score)]
                    c += 1
        sorted_scores = _sorted_scores

    _allowed = set([allowed for (allowed, _) in sorted_scores])
    for key in name_lines_dict.keys():
        if key not in _allowed:
            del name_lines_dict[key]

    sns.set(palette="Set2")

    names = [name for name in sorted(name_lines_dict.keys())]
    lines = [name_lines_dict[name] for name in sorted(name_lines_dict.keys())]

    data =  np.dstack(lines)

    if font_scale is not None:
        sns.set(font_scale=font_scale)
    sns.set_style(style='whitegrid')

    step = pd.Series(range(0, data.shape[1]), name='Epoch')
    #step = np.linspace(1, data.shape[1] +z 1, data.shape[1])
    types = pd.Series(names)

    cis = np.linspace(99, 95, 10, 4)
    ax = sns.tsplot(data, condition=types, time=step, value='Average Loss', ci=cis, err_style='ci_band',
                    interpolate=False, linewidth=1, color='husl', marker='x', markersize=12); # ci_bars, ci_band, boot_traces (standard error) # color='muted' # color="husl"

    print(ax.lines[-1])
    ax.lines[-1].set_marker('^')
    ax.lines[-2].set_marker('v')
    ax.lines[-3].set_marker('o')
    ax.lines[-4].set_marker('x')
    ax.legend()

    if loss_threshold is not None:
        ax.set_ylim(0, loss_threshold + 10)

    if title_str is not None:
        plt.title(title_str)

    if is_show:
        plt.show()
    else:
        sns.set_context('poster', font_scale=font_scale) # talk
        #plt.figure(figsize=(8, 6))

        fig = plt.gcf()
        fig.set_size_inches(10, 5) # (10, 6)

        save_file = 'out.png'
        if ('SAVE_FILE' in os.environ.keys()):
            save_file = os.environ['SAVE_FILE']

        fig.savefig(save_file, additional_artists=[], dpi=100, bbox_inches='tight')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
