#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import logging

import gzip

import util

def main(argv):
    entities_path = '../fb15k_entities.txt'

    with open(entities_path) as f:
        entities = set([entity.strip() for entity in f.readlines()])

    entity_to_iri, iri_to_entity = {}, {}
    for entity in entities:
        iri = util.entity_to_iri(entity)
        entity_to_iri[entity], iri_to_entity[iri] = iri, entity

    entity_iris = iri_to_entity.keys()

    entity_replacedBy = util.get_entity_replacedBy(entity_iris)
    for (k, v) in entity_replacedBy.items():
        entity_iris += v

    entity_iris = set(entity_iris)

    fb_path = '/home/datasets/freebase/old/freebase-rdf-2013-12-29-00-00.gz'

    count = 0
    with gzip.open(fb_path) as f:
        for line in f:

            count += 1
            if (count % 100000 == 0):
                print('%d' % (count), end='\r', file=sys.stderr)

            triple = line.split()
            if len(triple) >= 3:
                subject, predicate, object = triple[0][1:-1], triple[1][1:-1], triple[2][1:-1]
                if subject in entity_iris and object in entity_iris:
                    print(line.rstrip())

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
