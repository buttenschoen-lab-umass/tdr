#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np
from tdr.utils import asarray, round_to_nearest_fraction
from tdr.Boundary import DomainBoundary

from model.SimulationObject import SimulationObject
from model.SimulationObjectFactory import createSimObject
from model.xml_utils import isParameter


"""
    1-D only for the moment.
"""
class Interval(SimulationObject):
    def __init__(self, a = 0, b = 1, *args, **kwargs):
        # set the basic parameters
        self.name               = kwargs.pop('name', 'x')
        self.x0                 = a
        self.xf                 = b

        # number of patches -> in 1D always one
        self.nop                = 1

        # check if we have a xml node
        xml = kwargs.pop('xml', None)

        # Length parameters
        self.n                  = kwargs.pop('n', 8)
        self.cellsPerUnitLength = kwargs.pop('cellsPerUnitLength', 2**self.n)
        self.h                  = 1. / self.cellsPerUnitLength

        # for plotting
        self.y0                 = kwargs.pop('y0', 0.)
        self.yf                 = kwargs.pop('yf', 10.)

        self.bd                 = kwargs.pop('bd', DomainBoundary())

        if xml is not None:
            self._create_from_xml(xml, *args, **kwargs)

        # call reset
        self._reset()


    """ Creation """
    def _create_from_args(self, *args, **kwargs):
        pass


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
            else:
                assert False, 'Encountered unknown xml type %s' % child.tag

        # set the objects parameters!
        for p in parameters:
            setattr(self, p.name, p.value)


    """ Factory """
    class Factory:
        def create(self, *args, **kwargs):
            return Interval(*args, **kwargs)


    """ Internal methods """
    def _reset(self):
        self.L                  = np.abs(self.xf - self.x0)
        self.h                  = 1. / self.cellsPerUnitLength
        self.N                  = self.L * self.cellsPerUnitLength
        self.dX                 = asarray(self.h)
        self.x                  = self.xs()

        assert self.dX > 0, 'Interval must have non-negative step size.'
        assert self.N > 1, 'Interval N = %d must be larger than 1.' % self.N
        assert self.L > 0, 'Interval must have non-negative length.'


    """ Public methods """
    def origin(self):
        return np.asarray(self.x0)


    def dimensions(self):
        return 1


    def xs(self):
        return np.linspace(self.x0, self.xf - self.h, self.N)


    def box(self):
        return [self.x0, self.xf, self.y0, self.yf]


    def resize(self, arr):
        self.x0 = round_to_nearest_fraction(arr[0], self.n)
        self.xf = round_to_nearest_fraction(arr[1], self.n)
        N_old = self.N
        self._reset()
        return N_old


    def size(self):
        return int(self.N)


    def __repr__(self):
        return 'Interval(%.2f, %.2f, %d)' % (self.x0, self.xf, self.N)


    def __str__(self):
        return self.__repr__()


"""
    2-D square.
"""
class Square(object):
    def __init__(self, a, b, c, d, *args, **kwargs):
        # set the basic parameters
        self.x0                 = a
        self.xf                 = b
        self.y0                 = c
        self.yf                 = d

        # number of patches -> in 1D always one
        self.nop                = 1

        # Length parameters
        self.n                  = kwargs.pop('n', 8)
        self.cellsPerUnitLength = kwargs.pop('cellsPerUnitLength', 2**self.n)
        self.h                  = 1. / self.cellsPerUnitLength

        self.bd                 = kwargs.pop('bd', DomainBoundary())

        # call reset
        self._reset()


    """ Internal methods """
    def _reset(self):
        self.L          = np.array([self.xf - self.x0, self.yf - self.y0])
        self.N          = np.asarray([self.xf - self.x0, self.yf - self.y0]) * self.cellsPerUnitLength
        self.dX         = self.h * np.ones(2)


    """ Public methods """
    def origin(self):
        return np.asarray([self.x0, self.y0])


    def dimensions(self):
        return 2


    def xs(self):
        return np.linspace(self.x0, self.xf - self.h, self.N[0])


    def ys(self):
        return np.linspace(self.y0, self.yf - self.h, self.N[1])


    def box(self):
        return [self.x0, self.xf, self.y0, self.yf]


    def resize(self, arr):
        assert False, 'Not implemented for the 2D-square'


    def size(self):
        return int(np.sum(self.N))


    def __repr__(self):
        return 'Square(%.2f, %.2f, %.2f, %.2f, %d)' % (self.x0, self.xf, self.y0, self.yf, self.N)


    def __str__(self):
        return self.__repr__()


