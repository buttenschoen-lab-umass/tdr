#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np
from .utils import cartesian
from .Data import Data
from .Boundary import DomainBoundary2D
from .NonLocalGradient import NonLocalGradient


class Patch2D(object):
    """
        A patch of a domain.

        Parameters
        ----------
        n               : int
                          Number of PDEs
        patchId         : int
                          Id of the corresponding patch
        ngb             : dict
                          Identifying the boundary patches
        x0              : ndarray
                          The lower corner of the patch
        dX              : ndarray
                          Array of step lengths in each dimension
        N               : ndarray
                          Array giving size of partition set
        boundaryWidth   : int
                          Width of points to add for boundary conditions
        nonLocal        : bool
                          Whether a non-local gradient should be constructed
        **kwargs        : dict
                        These are forwarded to Data

        Data Layout
        -----------
        1) First the row of grid cells with the smallest x-values, then the row
           just above.

        2) Within each row of grid values start with the ones with the smallest
           y-values.


        Attributes
        ----------

        TODO

    """
    def __init__(self, n, patchId, x0, dX, N,
                 boundaryWidth = 0, nonLocal=False, **kwargs):
        self.x0  = x0
        self.N   = N
        self.n   = n
        self.dim = N.size
        self.dX  = dX
        self.ngb = kwargs.pop('ngb', DomainBoundary2D())
        self.shape = self.N * self.dX

        self.patchId = patchId
        # TODO set DEPRECATE ONE!
        self.boundaryWidth = boundaryWidth
        self.bW            = boundaryWidth
        self.nonLocalGradient = None

        # data for the patch
        self.data = None

        # array holding the cell centers
        self.xc = None

        # build cell centers
        self._setup_cell_centers()

        # setup
        self._setup(nonLocal, **kwargs)


    """ Public methods """
    def update(self, t, y):
        self.data.set_values(t, y)
        self.data.compute_face_data()


    def length(self):
        return self.N * self.dX


    def endPoints(self):
        return self.x0 + self.length()


    def apply_flux(self, flux):
        flux(self)


    def step_size(self):
        return self.dX


    def size(self):
        return np.prod(self.N)


    def cellCenters(self):
        return cartesian(self.xc)


    def get_ydot(self):
        return self.data.ydot


    """ Implementation details """
    def _setup_cell_centers(self):
        xcs = []
        for i in range(self.N.size):
            #print('i=',i,' N:', self.N, ' dX:', self.dX)
            xcs.append((np.arange(1, self.N[i] + 1, 1) - 0.5) * self.dX[i])

        self.xc = np.array(xcs)


    def _setup(self, nonLocal, **kwargs):
        if nonLocal:
            self._setup_nonlocal()

        self.data = Data(self.n, self.patchId, self.dX, self.boundaryWidth,
                         self.dim, ngb = self.ngb)


    def _setup_nonlocal(self):
        assert self.dX.size == 1, 'not implemented'
        self.nonLocalGradient = NonLocalGradient(self.dX[0], self.shape[0],
                                                 self.N[0])
