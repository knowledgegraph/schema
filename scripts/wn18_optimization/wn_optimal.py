#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

epochs_set, nbatches_set, ndim_set = [100], [10], [20]
lr_set = [00.000001, 00.000010, 00.000100, 00.001000, 00.010000, 00.100000, 01.000000, 10.000000]
decay_set = [0.9999, 0.9990, 0.9900, 0.9000, 0.5000]
epsilon_set = [0.000001, 0.001000]
max_lr_set = [10, 100, 1000]

seed_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# def sgd(param, rate, gradient, updates):
cmd_sgd = './learn_parameters.py --seed=%d --strategy=%s --totepochs=%d --lr=%f --ndim=%d  --name=wn_opt/wn_%s_%d  --train=%s --nbatches=%d --use_db --no_rescaling --sim=L1 --margin=2 > logs/wn_opt/wn.%s.%d.log 2>&1'

# def adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients):
cmd_adagrad = './learn_parameters.py --seed=%d --strategy=%s --totepochs=%d --lr=%f --ndim=%d  --name=wn_opt/wn_%s_%d  --train=%s --nbatches=%d --use_db --no_rescaling --sim=L1 --margin=2 > logs/wn_opt/wn.%s.%d.log 2>&1'

# def adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates):
cmd_adadelta = './learn_parameters.py --seed=%d --strategy=%s --totepochs=%d --epsilon=%f --decay=%f --ndim=%d  --name=wn_opt/wn_%s_%d  --train=%s --nbatches=%d --use_db --no_rescaling --sim=L1 --margin=2 > logs/wn_opt/wn.%s.%d.log 2>&1'

# def rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients):
cmd_rmsprop = './learn_parameters.py --seed=%d --strategy=%s --totepochs=%d --lr=%f --decay=%f --max_lr=%f --ndim=%d  --name=wn_opt/wn_%s_%d  --train=%s --nbatches=%d --use_db --no_rescaling --sim=L1 --margin=2 > logs/wn_opt/wn.%s.%d.log 2>&1'

# def momentum(param, rate, decay, gradient, updates, param_previous_update):
cmd_momentum = './learn_parameters.py --seed=%d --strategy=%s --totepochs=%d --lr=%f --decay=%f --ndim=%d  --name=wn_opt/wn_%s_%d  --train=%s --nbatches=%d --use_db --no_rescaling --sim=L1 --margin=2 > logs/wn_opt/wn.%s.%d.log 2>&1'

path = 'data/wn/WN-train.pkl'


# SGD
# def sgd(param, rate, gradient, updates):
c, method = 0, 'SGD'
for epochs in epochs_set:
    for nbatches in nbatches_set:
        for ndim in ndim_set:
            for lr in lr_set:
                for seed in seed_set:
                    print(cmd_sgd % (seed, method, epochs, lr, ndim, method, c, path, nbatches, method, c))
                    c += 1


# ADAGRAD
# def adagrad(param, rate, epsilon, gradient, updates, param_squared_gradients):
c, method = 0, 'ADAGRAD'
for epochs in epochs_set:
    for nbatches in nbatches_set:
        for ndim in ndim_set:
            for lr in lr_set:
                for seed in seed_set:
                    print(cmd_adagrad % (seed, method, epochs, lr, ndim, method, c, path, nbatches, method, c))
                    c += 1


# ADADELTA
# def adadelta(param, rate, decay, epsilon, gradient, updates, param_squared_gradients, param_squared_updates):
c, method = 0, 'ADADELTA'
for epochs in epochs_set:
    for nbatches in nbatches_set:
        for ndim in ndim_set:
            for epsilon in epsilon_set:
                for decay in decay_set:
                    for seed in seed_set:
                        print(cmd_adadelta % (seed, method, epochs, epsilon, decay, ndim, method, c, path, nbatches, method, c))
                        c += 1


# RMSPROP
# def rmsprop(param, rate, decay, max_learning_rate, epsilon, gradient, updates, param_squared_gradients):
c, method = 0, 'RMSPROP'
for epochs in epochs_set:
    for nbatches in nbatches_set:
        for ndim in ndim_set:
            for lr in lr_set:
                for decay in decay_set:
                    for max_lr in max_lr_set:
                        for seed in seed_set:
                            #print(cmd_rmsprop % (seed, method, epochs, lr, decay, max_lr, ndim, method, c, path, nbatches, method, c))
                            c += 1


# MOMENTUM
# def momentum(param, rate, decay, gradient, updates, param_previous_update):
c, method = 0, 'MOMENTUM'
for epochs in epochs_set:
    for nbatches in nbatches_set:
        for ndim in ndim_set:
            for lr in lr_set:
                for decay in decay_set:
                    for seed in seed_set:
                        print(cmd_momentum % (seed, method, epochs, lr, decay, ndim, method, c, path, nbatches, method, c))
                        c += 1
