#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from __future__ import print_function, division

from SimulationObject.SimulationObject import SimulationObject
from SimulationObject.SimulationObjectFactory import createSimObject
from utils.Scope import Scope
from utils.xml import isParameter
from model.ModelLogger import sim_logger


""" Simulation Object constructable from XML

    This is a temp name. TODO come up with a better abstraction
"""
class SimulationObjectXml(SimulationObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if we have a xml node
        xml = kwargs.pop('xml', None)
        if xml is not None:
            self._create_from_xml(xml, *args, **kwargs)
        else:
            assert False, 'Not implemented!'


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return SimulationObjectXml(*args, **kwargs)


    """ Process xml """
    def _create_from_xml(self, xml, *args, **kwargs):
        # set name
        setattr(self, 'name', xml.tag)

        # first check if the main node has attributes
        for name, value in xml.attrib.items():
            #assert not hasattr(self, name), 'Field: Attribute %s already registered!' % name
            setattr(self, name, value)

        # create scope
        self.scope = Scope(self.name, parent=kwargs.pop('parentScope', None))

        parameters = []
        for child in xml:
            p = createSimObject(child)
            if isParameter(child):
                parameters.append(p)
            else:
                assert False, 'Encountered unknown xml type %s' % child.tag

        # set the objects parameters!
        for p in parameters:
            self.scope.registerSymbol(p)






