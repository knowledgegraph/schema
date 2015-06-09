#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import cPickle as pickle
import logging


def merge(idx2entity_path, lhs_path, rel_path, rhs_path):
    idx2entity = pickle.load(open(idx2entity_path, 'rb'))
    lhs = pickle.load(open(lhs_path, 'rb'))
    rel = pickle.load(open(rel_path, 'rb'))
    rhs = pickle.load(open(rhs_path, 'rb'))

    rows, cols = rel.nonzero()
    logging.info((min(rows), max(rows)))

    #logging.info(len(idx2entity))

    entities = [idx2entity[i] for i in range(min(rows))]
    predicates = [idx2entity[i] for i in range(min(rows), max(rows) + 1)]

    logging.debug(len(entities))
    logging.debug(len(entities + predicates))

    all_1 = set(entities + predicates)
    all_2 = set([idx2entity[key] for key in idx2entity.keys()])

    logging.debug((all_2 - all_1, all_1 - all_2))

    obj = {
        'lhs': lhs,
        'rel': rel,
        'rhs': rhs,
        'entities': entities,
        'predicates': predicates
    }

    return obj


def main(argv):

    idx2entity_path = 'original/WN_idx2synset.pkl'

    train_lhs_path = 'original/WN-train-lhs.pkl'
    train_rel_path = 'original/WN-train-rel.pkl'
    train_rhs_path = 'original/WN-train-rhs.pkl'

    valid_lhs_path = 'original/WN-valid-lhs.pkl'
    valid_rel_path = 'original/WN-valid-rel.pkl'
    valid_rhs_path = 'original/WN-valid-rhs.pkl'

    test_lhs_path = 'original/WN-test-lhs.pkl'
    test_rel_path = 'original/WN-test-rel.pkl'
    test_rhs_path = 'original/WN-test-rhs.pkl'

    train_f = open('WN-train.pkl', 'wb')
    train = merge(idx2entity_path, train_lhs_path, train_rel_path, train_rhs_path)
    pickle.dump(train, train_f)
    train_f.close()

    valid_f = open('WN-valid.pkl', 'wb')
    valid = merge(idx2entity_path, valid_lhs_path, valid_rel_path, valid_rhs_path)
    valid['entities'] = train['entities']
    valid['predicates'] = train['predicates']
    pickle.dump(valid, valid_f)
    valid_f.close()

    test_f = open('WN-test.pkl', 'wb')
    test = merge(idx2entity_path, test_lhs_path, test_rel_path, test_rhs_path)
    test['entities'] = train['entities']
    test['predicates'] = train['predicates']
    pickle.dump(test, test_f)
    test_f.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
