# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import datetime, time
import sys, os, os.path
import pymongo
import cPickle as pickle
import logging


class PersistenceLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def create(self, name, document): pass

    @abstractmethod
    def get(self, id): pass

    @abstractmethod
    def update(self, id, document): pass

    @abstractmethod
    def remove(self, id): pass


class MongoLayer(PersistenceLayer):

    def __init__(self, url='mongodb://localhost:27017/energy', database='energy', collection='experiments'):
        self.client = pymongo.MongoClient(url)
        self.database = self.client[database]
        self.collection = self.database[collection]

    def create(self, name, document):
        document['name'] = name
        document_id = self.collection.insert(document)
        return document_id

    def get(self, id):
        document = self.collection.find_one(spec_or_id=id)
        return document

    def update(self, id, document):
        self.collection.update({ '_id': id }, document)

    def remove(self, id):
        self.collection.remove(id)


class PickleLayer(PersistenceLayer):

    def __init__(self, dir='~/models/', is_fast=False):
        self.dir = dir
        self.is_fast = is_fast

    def create(self, name, document):
        document['name'] = name
        timestr = time.strftime("%Y%m%d-%H%M%S")

        suffix, exists = 0, True;

        while exists:
            file_name = name + '_' + timestr + '_' + str(suffix) + '.pkl'
            self.path = os.path.expanduser(self.dir + '/' + file_name)
            exists = os.path.isfile(self.path)
            suffix += 1

        document['path'] = self.path
        self.__dump(self.path, document)
        return self.path

    def get(self, id):
        f = open(os.path.expanduser(id), 'rb')
        document = pickle.load(f)
        f.close()
        return document

    def update(self, id, document):
        self.__dump(id, document)

    def __dump(self, path, obj):
        fd = open(path, 'wb')
        if self.is_fast is True:
            protocol = pickle.HIGHEST_PROTOCOL
            p = pickle.Pickler(fd, protocol)
            p.fast = 1
            p.dump(obj)
        else:
            pickle.dump(obj, fd)
        fd.close()

    def remove(self, id):
        os.remove(id)
