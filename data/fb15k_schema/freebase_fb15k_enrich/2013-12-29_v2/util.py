#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import gzip
import rdflib

import refactoring

def entity_to_iri(entity):
    if '.' in entity:
        entity = entity.split('.')[1]
    return 'http://rdf.freebase.com/ns/' + entity.replace('/', '.')[1:]

def relation_to_iri(relation):
    relation_iri = 'http://rdf.freebase.com/ns/' + (relation.replace('/', '.')[1:])
    #print('%s -> %s' % (relation, relation_iri))
    return relation_iri

def get_entity_replacedBy(entity_iris):
    freebase_replacedBy_path = 'triples/replaced_by_sub.nt.gz'
    logging.debug('Importing replacedBy Relations ..')
    iri_to_replacedBy = {}
    with gzip.open(freebase_replacedBy_path) as f:
        for line in f:
            triple = line.split()
            if len(triple) > 3:
                subject, predicate, object = triple[0][1:-1], triple[1][1:-1], triple[2][1:-1]
                #if subject in entity_iris:
                if subject in iri_to_replacedBy:
                    iri_to_replacedBy[subject] += [object]
                else:
                    iri_to_replacedBy[subject] = [object]
                #if object in entity_iris:
                if object in iri_to_replacedBy:
                    iri_to_replacedBy[object] += [subject]
                else:
                    iri_to_replacedBy[object] = [subject]

    return iri_to_replacedBy

def get_domain_range(relations):
    freebase_dr_path = 'triples/domain_range.nt.gz'
    logging.debug('Importing Domain and Range Relations ..')

    dr_graph = rdflib.Graph()

    if freebase_dr_path.endswith('.gz'):
        with gzip.open(freebase_dr_path) as f:
            dr_graph.parse(f, format='nt')
    else:
        dr_graph.parse(freebase_dr_path, format='nt')
    iri_to_domains, iri_to_ranges = {}, {}

    for relation in relations:

        _relation = relation
        if _relation in refactoring.d:
            _relation = refactoring.d[_relation]

        relation_iri = relation_to_iri(_relation)

        tmp = _relation.split('.')

        first_relation = tmp[0]
        last_relation = tmp[-1]

        if first_relation in refactoring.d:
            first_relation = refactoring.d[first_relation]

        if last_relation in refactoring.d:
            last_relation = refactoring.d[last_relation]

        first_relation_iri = relation_to_iri(first_relation)
        last_relation_iri = relation_to_iri(last_relation)

        query_domain = ("""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?d WHERE { <%s> rdfs:domain ?d }
            """ % (first_relation_iri))

        query_range = ("""
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT DISTINCT ?r WHERE { <%s> rdfs:range ?r }
            """ % (last_relation_iri))

        qres_domain = dr_graph.query(query_domain)
        qres_range = dr_graph.query(query_range)

        relation_domains = [row['d'].n3()[1:-1] for row in qres_domain]
        relation_ranges = [row['r'].n3()[1:-1] for row in qres_range]

        if len(relation_domains) != 1 or len(relation_ranges) != 1:
            logging.warn('Relation: %s' % (relation_iri))

            logging.warn('First relation: %s' % (first_relation_iri))
            logging.warn('Last relation: %s' % (last_relation_iri))

        iri_to_domains[relation_iri] = relation_domains
        iri_to_ranges[relation_iri] = relation_ranges

    return (iri_to_domains, iri_to_ranges)

def get_entity_type(entity_iris, entity_replacedBy):
    freebase_type_path = 'triples/type_sub.nt.gz'
    _02p1pl6_path = 'triples/02p1pl6.nt.gz'
    logging.debug('Importing Type Relations ..')

    iri_to_types = {}

    for path in [freebase_type_path, _02p1pl6_path]:
        with gzip.open(path) as f:
            for line in f:
                triple = line.split()
                if len(triple) > 3:
                    # A is B
                    subject, predicate, object = triple[0][1:-1], triple[1][1:-1], triple[2][1:-1]

                    aliases = [subject]
                    if subject in entity_replacedBy:
                        aliases += entity_replacedBy[subject]

                    # if A, or one of its aliases, is in entity_iris:
                    for alias in set(aliases):
                        if alias in entity_iris:
                            # add the subject in iri_to_types
                            if alias in iri_to_types:
                                iri_to_types[alias] += [object]
                            else:
                                iri_to_types[alias] = [object]

    for entity_iri in entity_iris:
        entity_types = []
        if entity_iri in iri_to_types:
            entity_types = iri_to_types[entity_iri]
        if len(entity_types) == 0:
            logging.warn('Type for %s not found.' % (entity_iri))

    return iri_to_types

def enrich_types(entity_iri_to_types):
    ret = {}
    for (entity_iri, types) in entity_iri_to_types.items():
        if 'http://rdf.freebase.com/ns/medicine.risk_factor' in types:
            types += ['http://rdf.freebase.com/ns/medicine.disease_cause']
        if 'http://rdf.freebase.com/ns/film.production_company' in types:
            types += ['http://rdf.freebase.com/ns/film.film_company']
        if 'http://rdf.freebase.com/ns/business.job_title' in types:
            types += ['http://rdf.freebase.com/ns/people.profession']
        if 'http://rdf.freebase.com/ns/award.award_nominee' in types:
            types += ['http://rdf.freebase.com/ns/award.award_nominated_work']
        if 'http://rdf.freebase.com/ns/media_common.quotation_subject' in types:
            types += ['http://rdf.freebase.com/ns/tv.tv_genre']

        if 'http://rdf.freebase.com/ns/award.award' in types: #
            types += ['http://rdf.freebase.com/ns/time.recurring_event']

        if 'http://rdf.freebase.com/ns/user.jg.default_domain.olympic_games' in types:
            types += ['http://rdf.freebase.com/ns/olympics.olympic_games']
        if 'http://rdf.freebase.com/ns/user.jg.default_domain.olympic_sport' in types:
            types += ['http://rdf.freebase.com/ns/olympics.olympic_sport']

        if 'http://rdf.freebase.com/ns/olympics.olympic_games' in types:
            types += ['http://rdf.freebase.com/ns/user.jg.default_domain.olympic_games']
        if 'http://rdf.freebase.com/ns/olympics.olympic_sport' in types:
            types += ['http://rdf.freebase.com/ns/user.jg.default_domain.olympic_sport']

        if 'http://rdf.freebase.com/ns/base.aareas.schema.administrative_area' in types:
            types += ['http://rdf.freebase.com/ns/location.administrative_division']

        if 'http://rdf.freebase.com/ns/location.administrative_division' in types:
            types += ['http://rdf.freebase.com/ns/base.aareas.schema.administrative_area']

        if 'http://rdf.freebase.com/ns/media_common.netflix_title' in types:
            types += ['http://rdf.freebase.com/ns/film.film']

        if 'http://rdf.freebase.com/ns/location.citytown' in types:
            types += ['http://rdf.freebase.com/ns/film.film_screening_venue']

        if 'http://rdf.freebase.com/ns/media_common.adapted_work' in types:
            types += ['http://rdf.freebase.com/ns/film.film']

        if 'http://rdf.freebase.com/ns/base.dance.dance_form' in types:
            types += ['http://rdf.freebase.com/ns/music.genre']

        if 'http://rdf.freebase.com/ns/base.schemastaging.tv_program_extra' in types:
            types += ['http://rdf.freebase.com/ns/tv.tv_program']

        if 'http://rdf.freebase.com/ns/base.animemanga.anime_title' in types:
            types += ['http://rdf.freebase.com/ns/film.film']

        if 'http://rdf.freebase.com/ns/location.capital_of_administrative_division' in types:
            types += ['http://rdf.freebase.com/ns/location.administrative_division']

        if 'http://rdf.freebase.com/ns/people.profession' in types:
            types += ['http://rdf.freebase.com/ns/music.performance_role']

        if 'http://rdf.freebase.com/ns/music.group_member' in types:
            types += ['http://rdf.freebase.com/ns/music.artist']

        if 'http://rdf.freebase.com/ns/cvg.game_series' in types:
            types += ['http://rdf.freebase.com/ns/fictional_universe.fictional_universe']

        if 'http://rdf.freebase.com/ns/music.composer' in types:
            types += ['http://rdf.freebase.com/ns/music.artist']

        if 'http://rdf.freebase.com/ns/religion.religious_organization' in types:
            types += ['http://rdf.freebase.com/ns/religion.religion']

        if entity_iri == 'http://rdf.freebase.com/ns/m.0bkbz':
            types += ['http://rdf.freebase.com/ns/people.ethnicity']

        ret[entity_iri] = types
    return ret
