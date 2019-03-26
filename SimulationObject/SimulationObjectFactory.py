#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import generators, print_function

# NEED THESE!!
from .SimulationObject import SimulationObject


class SimulationObjectFactory:
    factories = {}

    @staticmethod
    def addFactory(type_name, simulationObjectFactory):
        SimulationObjectFactory.factories.put[type_name] = simulationObjectFactory

    @staticmethod
    def getSubClasses():
        types = SimulationObject.__subclasses__()
        for t in types:
            types.extend(t.__subclasses__())

        return types


    # A template method
    @staticmethod
    def createSimulationObject(type_name, *args, **kwargs):
        if not type_name in SimulationObjectFactory.factories:
            # get the type that we are requesting
            types = SimulationObjectFactory.getSubClasses()

            # lookup the factory method for type_name
            t = list(filter(lambda t : t.__name__ == type_name, types))
            assert len(t) > 0, 'Type %s is unknown.' % type_name
            SimulationObjectFactory.factories[type_name] = t[0].Factory()

        return SimulationObjectFactory.factories[type_name].create(*args, **kwargs)


def createSimObject(xml, *args, **kwargs):
    return SimulationObjectFactory.createSimulationObject(xml.tag, xml = xml, *args, **kwargs)


def createSimObjectByName(simObjectName, *args, **kwargs):
    return SimulationObjectFactory.createSimulationObject(simObjectName, *args, **kwargs)


