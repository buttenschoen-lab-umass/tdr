#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from __future__ import print_function, division

from SimulationObject.SimulationObject import SimulationObject
from SimulationObject.SimulationObjectFactory import createSimObject
from utils.Scope import Scope
from utils.xml import hasName
from parameter.Parameter import Parameter


""" Simulation Object constructable from XML

    This is a temp name. TODO come up with a better abstraction
"""
class SimulationObjectXml(SimulationObject):

    __xml_args__ = {
        Parameter.__name__ : 'registerParameterScope'
    }

    __par_registers__ = ['registerParameterScope',
                         'registerParameterScopeAttr',
                         'registerSimulationObjectScope',
                         'registerSimulationObject']


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if we have a xml node
        xml = kwargs.pop('xml', None)
        if xml is not None:
            self._create_from_xml(xml, *args, **kwargs)
        else:
            assert False, 'Not implemented!'


    """ Process xml """
    def _process_xml_attrib(self, attribs):
        for name, value in attribs.items():
            if name.lower() == 'requires':
                self.requirements.append(value)
            else:
                setattr(self, name, value)

        self.requirements = set(self.requirements)


    def _create_from_xml(self, xml, *args, **kwargs):
        # set name
        setattr(self, 'name', xml.tag)

        # first check if the main node has attributes
        self._process_xml_attrib(xml.attrib)

        # create scope
        self.scope = Scope(self.name, parent=kwargs.pop('parentScope', None))

        for child in xml:
            self.logger.info('Found xml entry: %s' % child.tag)
            sobj = createSimObject(child, parentScope=self.scope)

            # upon finding the right processor this must fail
            self.logger.debug('[%s] xml_args:' % self.name, self.__xml_args__)

            for cls_name, cls_registrar in self.__xml_args__.items():
                if hasName(child, cls_name):
                    self.logger.debug('[%s] found %s, %s.' % (self.name, child.tag, cls_name))
                    registrar = getattr(self, cls_registrar)

                    # We handle parameters and other different
                    if cls_registrar in self.__par_registers__:
                        # We have to register a parameter no destination required
                        self.logger.debug('[%s] parameter: %s.' % (self.name, sobj.name))
                        registrar(sobj)
                    else: # we have something other than a parameter
                        # if object has a short name use that!
                        if hasattr(sobj, '__short_name__'):
                            cls_name = sobj.__short_name__

                        dest = self.getDestinationName(cls_name, cls_registrar)
                        self.logger.debug('[%s] dest: %s.' % (self.name, dest))
                        registrar(dest, sobj)

                    break

            # If we got to the end of the above loop execute this
            else:
                # check if __xml_args__ has default key set if so add to simulation objects
                if 'default' in self.__xml_args__:
                    self.logger.debug('[%s] found %s, %s.' % (self.name, child.tag, cls_name))
                    #registrars = getattr(self, self.__xml_args__['default'])
                    for cls_registrar in self.__xml_args__['default']:
                        registrar = getattr(self, cls_registrar)
                        if cls_registrar in self.__par_registers__:
                            # We have to register a parameter no destination required
                            self.logger.debug('[%s] parameter: %s.' % (self.name, sobj.name))
                            registrar(sobj)
                        else:
                            assert False, 'Handling of \"%s\" does not exist!' % cls_registrar
                else:
                    assert False, 'Encountered unknown xml type %s' % child.tag


    """ register a Parameter in local scope """
    def registerParameterScope(self, p):
        self.logger.info('[%s] Registering: %s' % (self.name, p))
        self.scope.registerSymbol(p)


    """ register a Parameter in local scope and as class attribute """
    def registerParameterScopeAttr(self, p):
        self.logger.info('[%s] Registering: %s' % (self.name, p))
        self.scope.registerSymbol(p)
        setattr(self, p.name, p.value)


    """ get destination name """
    def getDestinationName(self, cls_name, method_name):
        method_name = method_name.lower()
        if 'list' in method_name:
            return cls_name.lower() + 's'
        elif 'attr' in method_name:
            return cls_name.lower()
        else:
            assert False, 'Unknown method type %s!' % method_name


    """ register sobj to local list """
    def registerSimulationObjectList(self, list_name, sobj):
        self.logger.debug('[%s] Registering: %s in list: %s' % (self.name, sobj.name, list_name))
        if not hasattr(self, list_name) or getattr(self, list_name) is None:
            self.logger.debug('List does not exist!')
            setattr(self, list_name, [])

        local_list = getattr(self, list_name)
        local_list.append(sobj)


    """ check that attribute isn't set yet """
    def _attribute_exists(self, attrib_name):
        assert (not hasattr(self, attrib_name) or getattr(self, attrib_name) is None), \
            '%s attribute already exists!' % attrib_name


    """ register sobj as local attribute """
    def registerSimulationObjectAttr(self, name, sobj):
        self._attribute_exists(name)
        self.logger.debug('[%s] Registering: %s as attribute!' % (self.name, name))
        setattr(self, name, sobj)


    """ register sobj in scope attribute """
    def registerSimulationObjectScope(self, sobj):
        self.logger.debug('Registering: %s in scope!' % sobj)
        self.scope.registerSymbol(sobj)


