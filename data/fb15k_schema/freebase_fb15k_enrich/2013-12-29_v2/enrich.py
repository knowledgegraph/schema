#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import logging

import gzip
import rdflib

import pickle
import util

import refactoring

def main(argv):
    entities_path = '../fb15k_entities.txt'
    relations_path = '../fb15k_relations.txt'

    with open(entities_path) as f:
        entities = set([entity.strip() for entity in f.readlines()])

    with open(relations_path) as f:
        relations = set([relation.strip() for relation in f.readlines()])

    entity_to_iri, iri_to_entity = {}, {}
    for entity in entities:
        iri = util.entity_to_iri(entity)
        entity_to_iri[entity], iri_to_entity[iri] = iri, entity

    relation_to_iri, iri_to_relation = {}, {}
    for relation in relations:
        _relation = relation
        if _relation in refactoring.d:
            _relation = refactoring.d[_relation]
        iri = util.relation_to_iri(_relation)
        relation_to_iri[relation], iri_to_relation[iri] = iri, relation

    entity_iris = set(iri_to_entity.keys())
    entity_replacedBy = util.get_entity_replacedBy(entity_iris)

    entity_iri_to_types = util.get_entity_type(entity_iris, entity_replacedBy)


    #entity_iri_to_types = util.enrich_types(entity_iri_to_types)


    (relation_iri_to_domains, relation_iri_to_ranges) = util.get_domain_range(relations)

    logging.info('Associating each Relation with two Entity Sets ..')

    relation_to_domain_entities = {}
    relation_to_range_entities = {}

    type_to_entity_iris = {}
    for (entity, types) in entity_iri_to_types.items():
        for type in types:
            if type in type_to_entity_iris:
                type_to_entity_iris[type] += [entity]
            else:
                type_to_entity_iris[type] = [entity]

    for relation in relations:
        relation_iri = relation_to_iri[relation]

        domain_types = relation_iri_to_domains[relation_iri]
        range_types = relation_iri_to_ranges[relation_iri]

        domain = []
        for domain_type in domain_types:
            if domain_type in type_to_entity_iris:
                domain += type_to_entity_iris[domain_type]

        range = []
        for range_type in range_types:
            if range_type in type_to_entity_iris:
                range += type_to_entity_iris[range_type]

        #print('%d %s %d' % (len(set(domain)), relation, len(set(range))))

        relation_to_domain_entities[relation] = set(domain)
        relation_to_range_entities[relation] = set(range)

    logging.info('Checking "illegal" triples in FB15k')

    with open('../freebase_mtr100_mte100.txt', 'r') as f:
        for line in f:
            triple = line.split()
            if len(triple) >= 3:
                subject, predicate, object = triple[0], triple[1], triple[2]

                domain = relation_to_domain_entities[predicate]
                range = relation_to_range_entities[predicate]

                if entity_to_iri[subject] not in domain or entity_to_iri[object] not in range:
                    #print(triple)
                    pass

                if subject not in entities or object not in entities:
                    logging.warn('subject %s or object %s not in entities' % (subject, object))
                if predicate not in relations:
                    logging.warn('relation %s not in relations' % (predicate))

    logging.info('Serializing domains and ranges ..')

    native_entity2idx = pickle.load(open('../idx/FB15k_entity2idx.pkl', 'rb'))

    NE, NP = len(entities), len(relations)

    entity2idx, relation2idx = {}, {}
    idx2entity, idx2relation = {}, {}

    for relation in relations:
        native_relation_idx = native_entity2idx[relation]
        relation_idx = native_relation_idx - NE

        relation2idx[relation] = relation_idx
        idx2relation[relation_idx] = relation

        logging.debug('Relation (%d): %s ' % (relation_idx, relation))

    for entity in entities:
        native_entity_idx = native_entity2idx[entity]
        entity_idx = native_entity_idx

        entity2idx[entity] = entity_idx
        idx2entity[entity_idx] = entity

        logging.debug('Entity (%d): %s ' % (entity_idx, entity))

    relation2domain, relation2range = {}, {}

    for relation in relations:
        relation_idx = relation2idx[relation]

        domain = relation_to_domain_entities[relation]
        range = relation_to_range_entities[relation]

        domain_idxs = [entity2idx[iri_to_entity[e]] for e in domain]
        range_idxs = [entity2idx[iri_to_entity[e]] for e in range]

        relation2domain[relation_idx] = domain_idxs
        relation2range[relation_idx] = range_idxs

    to_serialize = {
        'relation2domain': relation2domain,
        'relation2range': relation2range,

        'entity2idx': entity2idx,
        'relation2idx': relation2idx,

        'idx2entity': idx2entity,
        'idx2relation': idx2relation
    }

    f = open('fb15k_domains_ranges.pkl', 'wb')
    pickle.dump(to_serialize, f)
    f.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
