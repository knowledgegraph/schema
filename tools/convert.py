#!/usr/bin/python -uB
# -*- coding: utf-8 -*-

import sys
import getopt
import logging

import gzip
import rdflib


def convert(path, new_path):
    graph = rdflib.Graph()

    with gzip.open(path) as fd:
        graph.parse(fd, format='turtle')

    file = open(new_path, 'w')
    file.write(graph.serialize(format='nt'))

def main(argv):
    in_file, out_file = '/dev/stdin', '/dev/stdout'
    in_format, out_format = 'turtle', 'nt'

    usage_str = ('Usage: %s [-h] [--in_file=<file>] [--in_format=<format>] [--out_file=<file>] [--out_format=<format>]' % (sys.argv[0]))

    # Parse arguments
    try:
        opts, args = getopt.getopt(argv, 'h', ['in_file=', 'in_format=', 'out_file=', 'out_format='])
    except getopt.GetoptError:
        logging.warn(usage_str)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info(usage_str)

            logging.info('\t--in_file=<file> (default: %s)' % (in_file))
            logging.info('\t--in_format=<format> (default: %s)' % (in_format))

            logging.info('\t--out_file=<file> (default: %s)' % (out_file))
            logging.info('\t--out_format<format> (default: %s)' % (out_format))
            return

        if opt == '--in_file':
            in_file = arg
        if opt == '--in_format':
            in_format = arg
        if opt == '--out_file':
            out_file = arg
        if opt == '--out_format':
            out_format = arg

    graph = rdflib.Graph()

    logging.info('Parsing the input RDF Graph ..')

    if in_file.endswith('.gz'):
        with gzip.open(in_file) as fd:
            graph.parse(fd, format=in_format, encoding='utf-8')
    else:
        graph.parse(in_file, format=in_format, encoding='utf-8')

    logging.info('Serializing the RDF Graph ..')

    with open(out_file, 'w') as fd:
        fd.write(graph.serialize(format=out_format))

    logging.info('Closing the fd ..')

    fd.close()

    logging.info('Done.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(sys.argv[1:])
