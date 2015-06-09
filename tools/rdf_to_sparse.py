#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import getopt
import logging

import random

import json
import gzip
import cPickle as pickle
import rdflib

import numpy as np
import scipy.sparse as sp


def main(argv):
    kb_path, out_path, format = '/dev/stdin', 'out.pkl', 'nt'

    is_fast = False

    usage_str = ('Usage: %s [-h] [--kb=<file.nt>] [--format=<format>] [--out=<file.pkl>] [--fast]' % (sys.argv[0]))

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', ['kb=', 'format=', 'out=', 'fast'])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage_str)

            logging.info('\t--kb=<file.nt> (default: %s)' % (kb_path))
            logging.info('\t--format=<format> (default: %s)' % (format))
            logging.info('\t--out=<file.pkl> (default: %s)' % (out_path))

            logging.info('\t--fast (enable fast serialization, experimental)')
            return

        if opt == '--kb':
            kb_path = arg
        if opt == '--format':
            format = arg
        if opt == '--out':
            out_path = arg

        if opt == '--fast':
            is_fast = True

    logging.debug('Importing the RDF Knowledge Base from %s ..' % (kb_path))

    graph = rdflib.Graph()

    if kb_path.endswith('.gz'):
        with gzip.open(kb_path) as fd:
            graph.parse(fd, format=format)
    else:
        graph.parse(kb_path, format=format)

    logging.info('Extracting resources, blank nodes, predicates and literals ..')

    # extract a sorted list of resources and predicates from the set of RDF triples
    nodes = graph.all_nodes()
    predicates = sorted(set([p for (_, p, _) in graph]))
    triples = list(graph)

    # num. of triples, num. of resources, num. of predicates
    NT, NE, NP = len(graph), len(nodes), len(predicates)

    logging.debug('Number of triples in the RDF Graph: %d' % (NT))

    resources = [node for node in nodes if isinstance(node, rdflib.URIRef)]
    bnodes = [node for node in nodes if isinstance(node, rdflib.BNode)]
    literals = [node for node in nodes if isinstance(node, rdflib.Literal)]

    #for node in nodes:
    #    if (not isinstance(node, rdflib.URIRef)) and (not isinstance(node, rdflib.Literal)):
    #        logging.info('Node %s is type %s' % (node, type(node)))

    resources = sorted(set(resources))
    bnodes = sorted(set(bnodes))
    literals = list(set(literals))

    NR, NB, NL = len(resources), len(bnodes), len(literals)
    entities = resources + bnodes + literals

    logging.info('Resources: %d, BNodes: %d, Literals: %d (Total: %d), Nodes: %d:%d' % (NR, NB, NL, NR + NB + NL, NE, len(entities)))

    logging.info('Indexing entities and predicates ..')

    # dictionaries mapping each resource/predicate to its idx
    entity_to_idx = {entity:idx for (entity, idx) in zip(entities, range(NE))}
    predicate_to_idx = {predicate:idx for (predicate, idx) in zip(predicates, range(NP))}

    logging.debug('Serializing the RDF Knowledge Base in sparse matrices ..')

    # store the RDF graph as three sparse matrices, where the i-th triple is associated to the i-th column in each
    shape = (NE + NP, NT)
    inpl = sp.lil_matrix(shape, dtype='float32')
    inpr = sp.lil_matrix(shape, dtype='float32')
    inpo = sp.lil_matrix(shape, dtype='float32')

    # Populate the matrices
    for ct in range(NT):
        (s, p, o) = triples[ct]
        lhs = entity_to_idx[s]
        rel = NE + predicate_to_idx[p]
        rhs = entity_to_idx[o]
        inpl[lhs, ct], inpr[rhs, ct], inpo[rel, ct] = 1, 1, 1
        ct += 1

    logging.debug('Columns in the three sparse matrices: %d, %d, %d' % (int(inpl.sum()), int(inpr.sum()), int(inpo.sum())))
    logging.debug('Saving results ..')

    obj = {
        'lhs': inpl,
        'rhs': inpr,
        'rel': inpo,
        'entities': entities,

        'resources': resources,
        'bnodes': bnodes,
        'literals': literals,

        'predicates': predicates
    }

    fd = open(out_path, 'wb')

    if is_fast is True:

        _resources = ['R' + str(idx) for idx in range(len(resources))]
        _bnodes = ['B' + str(idx) for idx in range(len(bnodes))]
        _literals = ['L' + str(idx) for idx in range(len(literals))]
        _entities = _resources + _bnodes + _literals
        _predicates = ['P' + str(idx) for idx in range(len(predicates))]

        obj['resources'] = _resources
        obj['bnodes'] = _bnodes
        obj['literals'] = _literals
        obj['entities'] = _entities
        obj['predicates'] = _predicates

        protocol = pickle.HIGHEST_PROTOCOL
        p = pickle.Pickler(fd, protocol)
        p.fast = 1
        p.dump(obj)
    else:
        pickle.dump(obj, fd)

    fd.close()

    logging.debug('Done.')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
