#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#

from __future__ import print_function
import numpy as np

from model.SimulationObject import SimulationObject
from model.SimulationObjectFactory import createSimObject, createSimObjectByName
from model.xml_utils import isParameter, isBoundary


"""
    Keeps information that are important about choosing a type of boundary
    condition for a face.
"""
class Boundary(SimulationObject):
    def __init__(self, name = "Boundary", oNormal = 1., *args, **kwargs):
        self.name = name

        # TODO: is there a better way to do this?
        self.bc_lookup = {'None' : 0, 'Periodic' : 1, 'Neumann' : 2,
                          'Dirichlet' : 3, 'NoFlux' : 4}

        # default value for value and oNormal
        self.value = 1.

        # the output normal
        self.oNormal = np.asarray(oNormal)

        xml = kwargs.pop('xml', None)
        if xml is not None:
            self._create_from_xml(xml, *args, **kwargs)

        # setup
        self._setup()


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return Boundary(*args, **kwargs)


    """ xml factory """
    def _create_from_xml(self, xml, *args, **kwargs):
        # set name
        setattr(self, 'name', kwargs.pop('type', 'Periodic'))


    """ private methods """
    def _validate(self):
        assert self.name in self.bc_lookup, 'Unknown boundary condition type %s!' % self.name
        assert np.linalg.norm(self.oNormal) == 1., 'oNormal has to be a unit vector!'


    def _setup(self):
        assert self.name is not "Boundary", 'Boundary type must be set!'
        self._validate()
        self.type = self.bc_lookup[self.name]
        print('Registered %s boundary.' % self.name)


    def __str__(self):
        return self.name


    """ public methods """
    def isPeriodic(self):
        return self.type == self.bc_lookup["Periodic"]


    def isNeumann(self):
        return self.type == self.bc_lookup["Neumann"]


    def isNoFlux(self):
        return self.type == self.bc_lookup["NoFlux"]


    def isDirichlet(self):
        return self.type == self.bc_lookup["Dirichlet"]


    def __call__(self):
        return self.value


class Periodic(Boundary):
    def __init__(self):
        super(Periodic, self).__init__(name = "Periodic")


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return Periodic(*args, **kwargs)


class Neumann(Boundary):
    def __init__(self, oNormal, value = 0.):
        super(Neumann, self).__init__(name = "Neumann", oNormal = oNormal)
        self.value = value


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return Neumann(*args, **kwargs)


class NoFlux(Boundary):
    def __init__(self, oNormal, value = 0.):
        super(NoFlux, self).__init__(name = "NoFlux", oNormal = oNormal)
        self.value = value


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return NoFlux(*args, **kwargs)


class Dirichlet(Boundary):
    def __init__(self, oNormal, value = 0.):
        super(Dirichlet, self).__init__(name = "Dirichlet", oNormal = oNormal)
        self.value = value


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return Dirichlet(*args, **kwargs)


"""
    This class implements easy setup of boundary conditions
"""

""" 1D only at the moment """
class DomainBoundary(SimulationObject):
    def __init__(self, *args, **kwargs):
        # spatial dimension
        self.dim = kwargs.pop('dim', 1)

        xml = kwargs.pop('xml', None)
        if xml is not None:
            self._create_from_xml(xml, *args, **kwargs)
        else:
            if self.dim == 1:
                self._setup_1d(*args, **kwargs)
            elif self.dim == 2:
                self._setup_2d(*args, **kwargs)
            else:
                assert 'DomainBoundary not implemented for %dd' % self.dim


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return DomainBoundary(*args, **kwargs)


    """ xml factory """
    def _create_from_xml(self, xml, *args, **kwargs):
        # set name
        setattr(self, 'name', xml.tag)

        # first check if the main node has attributes
        for name, value in xml.attrib.items():
            setattr(self, name, value)

        parameters = []
        for child in xml:
            p = createSimObject(child)
            if isParameter(child):
                parameters.append(p)
            elif isBoundary(child):
                attributes = child.attrib
                setattr(self, attributes['where'], p)
            else:
                assert False, 'Encountered unknown xml type %s' % child.tag


        # set the objects parameters!
        for p in parameters:
            setattr(self, p.name, p.value)


    """ setup methods """
    def _setup_1d(self, *args, **kwargs):
        # process arguments
        self.left  = kwargs.pop('left',  Neumann(-1.))
        self.right = kwargs.pop('right', Neumann( 1.))

        # allow easy lookup
        setattr(self, 'left', self.left)
        setattr(self, 'right', self.right)

        # easy indexing
        self.lookup = {0 : self.left, 1 : self.right}


    def _setup_2d(self, *args, **kwargs):
        # process arguments
        self.left  = kwargs.pop('left',  Neumann(-1.))
        self.right = kwargs.pop('right', Neumann( 1.))

        self.top    = kwargs.pop('top',    Neumann(-1.))
        self.bottom = kwargs.pop('bottom', Neumann( 1.))

        # allow easy lookup
        setattr(self, 'left',   self.left)
        setattr(self, 'right',  self.right)
        setattr(self, 'top',    self.top)
        setattr(self, 'bottom', self.bottom)

        # easy indexing
        self.lookup = {0 : self.left, 1 : self.right, 2 : self.bottom, 3 : self.top}


    """ internal methods """
    def __getitem__(self, key):
        return self.lookup[key]


    def __iter__(self):
        for bd in self.lookup.values():
            yield bd


    """ String """
    def __str__(self):
        return "DomainBoundary"


    def __repr__(self):
        return "DomainBoundary"


    """ public methods """
    # TODO: atm we can't deal with different boundary conditions on either side
    def isPeriodic(self):
        return self.left.isPeriodic() and self.right.isPeriodic()


    def isNeumann(self):
        return self.left.isNeumann() and self.right.isNeumann()


    def isNoFlux(self):
        return self.left.isNoFlux() and self.right.isNoFlux()


    def isDirichlet(self):
        return self.left.isDirichlet() and self.right.isDirichlet()


if __name__ == '__main__':
    print('test')

    bc = Neumann()
    print(bc)
    print(bc.isNeumann())
    print(bc.isDirichlet())


