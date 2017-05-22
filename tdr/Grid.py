#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np
from Patch import Patch


class Grid(object):
    def __init__(self, n, grd, dim, bw = 0, nonLocal=False):
        # number of concentration fields
        self.n = n

        ## dimensions
        self.dim = dim

        # boundary width
        self.boundaryWidth = bw

        # cell centers
        self.cellCentreMatrix = None

        # TODO needed?
        self.molsize  = 0
        self.gridsize = 0

        # map to patches
        self.patches = {}

        # setup
        self._setup(grd, nonLocal)


    """ Public methods """
    def patches(self):
        return self.patches


    def update(self, t, y):
        # TODO need to unwrap y
        for patch in self.patches.values():
            patch.update(t, y)


    def apply_flux(self, flux):
        for patch in self.patches.values():
            patch.apply_flux(flux)


    def get_dx(self, patchId):
        return self.patches[patchId].dX


    def get_nx(self, patchId):
        return self.patches[patchId].N


    def get_start(self, patchId):
        return self.ps[patchId]


    def get_end(self, patchId):
        return self.pe[patchId]


    """ Implementation details """
    def _setup(self, grd, nonLocal):
        self._init_patches(grd, nonLocal)
        self._compute_size()
        self._compute_cellCenter()


    def _init_patches(self, grd, nonLocal):
        nop = grd['nop']
        ngb = grd['ngb']
        dX  = grd['dX']
        N   = grd['N']
        x0  = grd['x0']

        for i in range(nop):
            self.patches[i] = Patch(self.n, i + 1, x0[i], dX[i], N[i],
                                    ngb             = ngb[i],
                                    boundaryWidth   = self.boundaryWidth,
                                    nonLocal        = nonLocal)


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





