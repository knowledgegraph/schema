#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import cPickle as pickle
import logging


def merge(dataset, idx2entity_path, lhs_path, rel_path, rhs_path, targets_path=None):
    idx2entity = pickle.load(open(idx2entity_path, 'rb'))

    lhs = pickle.load(open(lhs_path, 'rb'))
    rel = pickle.load(open(rel_path, 'rb'))
    rhs = pickle.load(open(rhs_path, 'rb'))

    if dataset == 'WN11':
        Nent = 38588
        Nrel = 11
    else:
        raise ValueError('Unknown dataset: %s' % (dataset))

    entities = [idx2entity[i] for i in range(Nent)]
    predicates = [idx2entity[i] for i in range(Nent, Nent + Nrel)]

    logging.info('#Entities: %d' % (len(entities)))
    logging.info('#Predicates: %d' % (len(predicates)))

    specs = {
        'Nent': Nent,
        'Nrel': Nrel
    }

    obj = {
        'lhs': lhs,
        'rel': rel,
        'rhs': rhs,

        'entities': entities,
        'predicates': predicates,

        'specs': specs
    }

    if targets_path is not None:
        targets = pickle.load(open(targets_path, 'rb'))
        obj['targets'] = targets

    return obj


def serialize(obj, path):
    f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()


def main(argv):

    for dataset in ['WN11']:
        idx2entity_path = 'data/%s_idx2entity.pkl' % dataset

        train_lhs_path = 'data/%s-train-lhs.pkl' % dataset
        train_rel_path = 'data/%s-train-rel.pkl' % dataset
        train_rhs_path = 'data/%s-train-rhs.pkl' % dataset

        valid_lhs_path = 'data/%s-valid-lhs.pkl' % dataset
        valid_rel_path = 'data/%s-valid-rel.pkl' % dataset
        valid_rhs_path = 'data/%s-valid-rhs.pkl' % dataset
        valid_targets_path = 'data/%s-valid-targets.pkl' % dataset

        test_lhs_path = 'data/%s-test-lhs.pkl' % dataset
        test_rel_path = 'data/%s-test-rel.pkl' % dataset
        test_rhs_path = 'data/%s-test-rhs.pkl' % dataset
        test_targets_path = 'data/%s-test-targets.pkl' % dataset

        train = merge(dataset, idx2entity_path, train_lhs_path, train_rel_path, train_rhs_path)
        serialize(train, '%s-train.pkl' % dataset)

        valid = merge(dataset, idx2entity_path, valid_lhs_path, valid_rel_path, valid_rhs_path, targets_path=valid_targets_path)
        serialize(valid, '%s-valid.pkl' % dataset)

        test = merge(dataset, idx2entity_path, test_lhs_path, test_rel_path, test_rhs_path, targets_path=test_targets_path)
        serialize(test, '%s-test.pkl' % dataset)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
