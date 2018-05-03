#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np
from .utils import cartesian
from .Data import Data
from .NonLocalGradient import NonLocalGradient
from .Boundary import DomainBoundary


class Patch(object):
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


        Attributes
        ----------

        TODO

    """
    def __init__(self, n, patchId, x0, dX, N, boundaryWidth = 0, nonLocal=False, **kwargs):
        print('Creating patch: %d, origin: %s, dX: %s, N: %s.' % (patchId, x0, dX, N))

        self.x0  = x0
        self.N   = N
        self.n   = n
        self.dim = N.size
        self.dX  = dX
        self.ngb = kwargs.pop('ngb', DomainBoundary(self.dim))
        self.shape = self.N * self.dX

        # TODO: Solve this more elegantly
        if nonLocal and (self.ngb.left.name != 'Periodic' or self.ngb.right.name != 'Periodic'):
            assert False, 'Non-local equation can only be solved on a periodic domain!'

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


    def get_shape(self):
        return self.N


    """ Resizing of the domain """
    # only for 1D so far
    # TODO: optimize the recomp of cell centers!
    def growPatchRight(self):
        # easy simply add N
        self.N += 1

        # redo cell centers
        self._setup_cell_centers()


    def growPatchLeft(self):
        # grow to the right but first move left most point
        self.x0 -= self.dX
        self.growPatchRight()


    def shrinkPatchRight(self):
        # reduce N by one
        self.N -= 1
        self._setup_cell_centers()


    def shrinkPatchLeft(self):
        # shift x0 to the right
        self.x0 += self.dX
        self.shrinkPatchRight()


    """ Implementation details """
    def _setup_cell_centers(self):
        xcs = []
        for i in range(self.N.size):
            print('i=',i,' N:', self.N, ' dX:', self.dX)
            xcs.append((np.arange(1, self.N[i] + 1, 1) - 0.5) * self.dX[i])

        self.xc = np.array(xcs)


    def _setup(self, nonLocal, **kwargs):
        if nonLocal:
            self._setup_nonlocal()

        self.data = Data(self.n, self.patchId, self.dX, self.boundaryWidth,
                         self.dim, ngb = self.ngb)


    def _setup_nonlocal(self):
        assert self.dX.size == 1, 'not implemented'
        self.nonLocalGradient = NonLocalGradient(self.dX[0], self.shape[0], self.N[0])


