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

        # patch start
        self.ps = {}
        # patch end
        self.pe = {}

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


    def get_ydot(self):
        ret = np.zeros((self.n * self.gridsize))
        for patchId, patch in self.patches.items():
            # TODO make more general
            pstart = self.ps[patchId]
            pend   = self.pe[patchId]
            ret[pstart:pend] = patch.get_ydot().flatten()

        return ret


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

            # TODO make more general!
            self.ps[i] = 0
            self.pe[i] = int(self.n * N[i])


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





