#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function

import numpy as np
from tdr.Patch import Patch
from tdr.helpers import expand_dims


class Grid(object):
    def __init__(self, n, grd, dim, *args, **kwargs):
        # number of concentration fields
        self.n = n

        ## dimensions
        self.dim = dim

        # boundary width
        self.boundaryWidth = kwargs.pop('bw', 0)

        # cell centers
        self.cellCentreMatrix = None

        # TODO needed?
        self.molsize  = 0
        self.gridsize = 0

        # map to patches
        self.patches = {}

        # patch start
        self.ps = {}
        # patch end
        self.pe = {}

        # setup
        self._setup(grd, *args, **kwargs)


    """ Public methods """
    def patches(self):
        return self.patches


    def __repr__(self):
        return 'Grid(n = %d; dim = %d; patches = %d; bw = %d)' % \
                (self.n, self.dim, len(self.patches), self.boundaryWidth)


    def __str__(self):
        return self.__repr__()


    """ update grid information """
    def update(self, t, y):
        # TODO need to unwrap y
        for patch in self.patches.values():
            # The patch update function has to run before any deformations!
            patch.update(t, y)


    """ Apply fluxes for all patches """
    def apply_flux(self, flux, t):
        for patch in self.patches.values():
            patch.apply_flux(flux, t)


    def get_dx(self, patchId):
        return self.patches[patchId].dX


    def get_nx(self, patchId):
        return self.patches[patchId].N


    def get_start(self, patchId):
        return self.ps[patchId]


    def get_end(self, patchId):
        return self.pe[patchId]


    def get_ydot(self):
        ret = np.zeros((self.n * self.gridsize))
        for patchId, patch in self.patches.items():
            # TODO make more general
            pstart = self.ps[patchId]
            pend   = self.pe[patchId]
            ret[pstart:pend] = patch.get_ydot().flatten()

        return ret


    def elongate(self, where, direction):
        # TODO fill in
        pass


    def shape(self):
        shp = np.zeros(self.dim, dtype=np.int)
        for patch in self.patches.values():
            shp += patch.get_shape()

        return shp


    """ Implementation details """
    def __iter__(self):
        for patch in self.patches.values():
            yield patch


    def __getitem__(self, key):
        return self.patches[key]


    """ Setup any required internal data structures """
    def _setup(self, grd, *args, **kwargs):
        nonLocal = kwargs.pop('nonLocal', False)
        self._init_patches(grd, nonLocal, *args, **kwargs)

        # Setup internal data structures
        self._compute_size()
        self._compute_cellCenter()


    def _init_patches(self, grd, nonLocal, *args, **kwargs):
        nop = grd['nop']
        ngb = grd['ngb']
        dX  = grd['dX']
        N   = grd['N']
        x0  = grd['x0']

        if nop == 1: # and isinstance(ngb, DomainBoundary):
            # expand dims if required
            dX = expand_dims(dX, self.dim)
            N  = expand_dims(N,  self.dim)
            x0 = expand_dims(x0, self.dim)

        for i in range(nop):
            self.patches[i] = Patch(self.n, i + 1, x0[i], dX[i], N[i],
                                    ngb             = ngb[i],
                                    boundaryWidth   = self.boundaryWidth,
                                    nonLocal        = nonLocal,
                                    *args, **kwargs)

            # TODO make more general!
            self.ps[i] = 0
            self.pe[i] = int(self.n * np.prod(N[i]))


    def _compute_size(self):
        self.gridsize = 0
        for patchId, patch in self.patches.items():
            self.gridsize += patch.size()


    def _compute_cellCenter(self):
        self.cellCentreMatrix = np.zeros((self.gridsize, self.dim))
        offset = 0
        for patchId, patch in self.patches.items():
            self.cellCentreMatrix = np.insert(self.cellCentreMatrix,
                                              offset, patch.cellCenters(),
                                              axis=0)


if __name__ == '__main__':
    cellsPerUnitLength = 10
    h = 1. / cellsPerUnitLength
    L = 1.

    nop = 1
    ngb = np.array([[1, 1, 1, 1]])
    dX  = np.array([[h, h]])
    x0  = np.array([[0, 0]])
    N   = np.array([[L * cellsPerUnitLength, L * cellsPerUnitLength]]).astype(int)

    grd = { 'nop' : nop, 'ngb' : ngb, 'dX' : dX, 'N' : N, 'x0' : x0}

    n = 2
    grid = Grid(n, grd, 2)





