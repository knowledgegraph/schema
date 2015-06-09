#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import getopt
import logging

import random

import json
import gzip
import rdflib

import numpy as np
import scipy.sparse as sp

def main(argv):
    kb_path, format, block_size = '/dev/stdin', 'nt', 10000
    train_path, valid_path, test_path = 'train.nt', 'valid.nt', 'test.nt'

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', ['kb=', 'format=', 'blocksize=', 'train=', 'valid=', 'test='])
    except getopt.GetoptError:
        logging.warn('Usage: %s [-h] [--kb=<file.nt>] [--format=<format>] [--train=<train.nt>] [--valid=<valid.nt>] [--test=<test.nt>]' % (sys.argv[0]))
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info('Usage: %s [-h] [--kb=<file.nt>] [--format=<format>] [--train=<train.nt>] [--valid=<valid.nt>] [--test=<test.nt>]' % (sys.argv[0]))

            logging.info('\t-h')

            logging.info('\t--kb=<file.nt> (default: %s)' % (kb_path))
            logging.info('\t--format=<format> (default: %s)' % (format))
            logging.info('\t--blocksize=<block_size> (default: %s)' % (block_size))

            logging.info('\t--train=<train.nt> (default: %s)' % (train_path))
            logging.info('\t--valid=<valid.nt> (default: %s)' % (valid_path))
            logging.info('\t--test=<test.nt> (default: %s)' % (test_path))

            return
        if opt == '--kb':
            kb_path = arg
        if opt == '--format':
            format = arg
        if opt == '--blocksize':
            block_size = int(arg)

        if opt == '--train':
            train_path = arg
        if opt == '--valid':
            valid_path = arg
        if opt == '--test':
            test_path = arg

    logging.debug('Importing the RDF Graph ..')

    graph = rdflib.Graph()

    if kb_path.endswith('.gz'):
        with gzip.open(kb_path) as fd:
            graph.parse(fd, format=format)
    else:
        graph.parse(kb_path, format=format)

    triples = list(graph)
    NT = len(triples)

    logging.debug('Number of total unique triples: %s' % (len(set(triples))))

    logging.debug('Generating a random permutation of RDF triples ..')

    # set the random seed
    random.seed(666)

    # generate a random permutation for the set of triples
    permutation = range(NT)
    random.shuffle(permutation)

    logging.debug('Building the training set ..')
    train = [triples[permutation[i]] for i in range(0, NT - (block_size * 2))]
    logging.debug('Number of triples: %s' % (len(train)))
    train_graph = rdflib.Graph()
    for train_triple in train:
        train_graph.add(train_triple)
    train_graph.serialize(destination=train_path, format=format)

    logging.debug('Building the validation set ..')
    valid = [triples[permutation[i]] for i in range(NT - (block_size * 2), NT - block_size)]
    logging.debug('Number of triples: %s' % (len(valid)))
    valid_graph = rdflib.Graph()
    for valid_triple in valid:
        valid_graph.add(valid_triple)
    valid_graph.serialize(destination=valid_path, format=format)

    logging.debug('Building the test set ..')
    test = [triples[permutation[i]] for i in range(NT - block_size, NT)]
    logging.debug('Number of triples: %s' % (len(test)))
    test_graph = rdflib.Graph()
    for test_triple in test:
        test_graph.add(test_triple)
    test_graph.serialize(destination=test_path, format=format)

    logging.debug('Number of total (train + valid + test) unique triples: %s' % (len(set(train + valid + test))))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
