#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
#

from __future__ import print_function
from numbers import Number
import numpy as np

"""
    This class implements easy setup of boundary conditions
"""

""" 1D only at the moment """
class DomainBoundary(object):
    def __init__(self, left, right):
        self.left  = left
        self.right = right

        # allow easy lookup
        setattr(self, 'left', self.left)
        setattr(self, 'right', self.right)
        self.list = [self.left, self.right]


    """ internal methods """
    def __getitem__(self, key):
        if key == 0:
            return self.left
        elif key == 1:
            return self.right
        else:
            assert False, 'DomainBoundary unknown key'


    def __iter__(self):
        for bd in self.list:
            yield bd


    """ public methods """
    # TODO: atm we can't deal with different boundary conditions on either side
    def isPeriodic(self):
        return self.left.isPeriodic() and self.right.isPeriodic()


    def isNeumann(self):
        return self.left.isNeumann() and self.right.isNeumann()


    def isDirichlet(self):
        return self.left.isDirichlet() and self.right.isDirichlet()


"""
    Keeps information that are important about choosing a type of boundary
    condition for a face.
"""
class Boundary(object):
    def __init__(self, name = "Boundary", oNormal = np.array([1.])):
        self.name = name

        # TODO: is there a better way to do this?
        self.bc_lookup = {'None' : 0, 'Periodic' : 1, 'Neumann' : 2, 'Dirichlet' : 3}

        # default value for value and oNormal
        self.value = 1.

        if isinstance(oNormal, Number):
            self.oNormal = np.array([oNormal])
        else:
            self.oNormal = oNormal

        # setup
        self._setup()


    """ private methods """
    def _validate(self):
        assert self.name in self.bc_lookup, 'Unknown boundary condition type %s!' % self.name
        assert np.linalg.norm(self.oNormal) == 1., 'oNormal has to be a unit vector!'


    def _setup(self):
        self._validate()
        self.type = self.bc_lookup[self.name]


    def __str__(self):
        return self.name


    """ public methods """
    def isPeriodic(self):
        return self.type == self.bc_lookup["Periodic"]


    def isNeumann(self):
        return self.type == self.bc_lookup["Neumann"]


    def isDirichlet(self):
        return self.type == self.bc_lookup["Dirichlet"]


    def __call__(self):
        return self.value


class Periodic(Boundary):
    def __init__(self):
        super(Periodic, self).__init__(name = "Periodic")


class Neumann(Boundary):
    def __init__(self, value, oNormal):
        super(Neumann, self).__init__(name = "Neumann", oNormal = oNormal)
        self.value = value


class Dirichlet(Boundary):
    def __init__(self, value, oNormal):
        super(Dirichlet, self).__init__(name = "Dirichlet", oNormal = oNormal)
        self.value = value


if __name__ == '__main__':
    print('test')

    bc = Neumann()
    print(bc)
    print(bc.isNeumann())
    print(bc.isDirichlet())


