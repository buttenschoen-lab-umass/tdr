#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function

import numpy as np
from tdr.Patch import Patch
from tdr.helpers import expand_dims


class Grid(object):
    def __init__(self, n, domain, *args, **kwargs):
        # number of concentration fields
        self.n = n

        ## dimensions
        self.dim = domain.dimensions

        # boundary width
        self.boundaryWidth = kwargs.pop('bw', 0)

        # cell centers
        self.cellCentreMatrix = None

        # molsize is the size of the y - vector
        self.molsize  = 0

        # gridsize is the total number of discretization volumes
        self.gridsize = 0

        # map to patches
        self.patches = []

        # the main memory for ydot
        self.ydot = None

        # setup
        self._setup(domain, *args, **kwargs)


    """ Public methods """
    def patches(self):
        return self.patches


    @property
    def dy(self):
        return self.reshape(self.ydot)


    def __repr__(self):
        return 'Grid(n = %d; dim = %d; patches = %d; bw = %d)' % \
                (self.n, self.dim, len(self.patches), self.boundaryWidth)


    def __str__(self):
        return self.__repr__()


    """ update grid information """
    def update(self, t, y):
        for patch in self.patches:
            # The patch update function has to run before any deformations!
            patch.update(t, y)


    """ Apply fluxes for all patches """
    def apply_flux(self, flux, t):
        for patch in self.patches:
            patch.apply_flux(flux, t)


    def get_dx(self, patchId):
        return self.patches[patchId].dX


    def get_nx(self, patchId):
        return self.patches[patchId].N


    def get_ydot(self):
        return self.ydot


    def elongate(self, where, direction):
        assert False, 'Not implemented!'


    def shape(self):
        shp = np.zeros(self.dim, dtype=np.int)
        for patch in self.patches:
            shp += patch.get_shape()

        return shp


    """ Implementation details """
    def __iter__(self):
        for patch in self.patches:
            yield patch


    def __getitem__(self, key):
        return self.patches[key]


    """ Setup any required internal data structures """
    def _setup(self, domain, *args, **kwargs):
        self._init_patches(domain, *args, **kwargs)

        # Setup internal data structures
        self._compute_size()
        self._compute_cellCenter()

        # setup ydot
        self.ydot = np.zeros(self.molsize)

        # assign memory to patches
        self._assign_memory_patches()


    def _init_patches(self, domain, *args, **kwargs):
        # By default assuming we do not require non-local operator support
        nonLocal = kwargs.pop('nonLocal', False)

        nop = domain.nop
        N   = domain.N
        x0  = domain.x0s
        xf  = domain.xfs
        ngb = domain.bds
        dX  = domain.dX

        offset = 0
        for i in range(nop):
            dX_i = expand_dims(dX[i], self.dim)
            N_i  = expand_dims(N[i],  self.dim)
            x0_i = expand_dims(x0[i], self.dim)
            xf_i = expand_dims(xf[i], self.dim)

            patch = Patch(n=self.n, start=offset, x0=x0_i, xf=xf_i, dX=dX_i, N=N_i,
                          patchId=i+1, ngb=ngb[i], boundaryWidth=self.boundaryWidth,
                          nonLocal=nonLocal, *args, **kwargs)

            # define locations of a patch in the general y - vector
            offset += patch.size()

            # save patch
            self.patches.append(patch)


    """ Give each patch access to its component of the ydot vector """
    def _assign_memory_patches(self):
        ydot = self.reshape(self.ydot)
        for patch in self.patches:
            patch.set_ydot(ydot[:, patch.ps:patch.pe])


    """ reshape a 1D-vector to the correct 2D-shape """
    def reshape(self, y):
        return y.reshape((self.n, self.gridsize))


    """ Compute the Grids physical size and memory size """
    def _compute_size(self):
        self.domsize  = np.zeros(self.dim).astype(np.int)
        self.gridsize = 0
        self.molsize  = 0
        for patch in self.patches:
            self.gridsize += patch.size()
            self.domsize  += patch.grid_size()
            self.molsize  += patch.memory_size()


    """ Computes the centers for the whole grid """
    def _compute_cellCenter(self):
        nshape = tuple(self.domsize) + (self.dim,)
        self.cellCentreMatrix = np.zeros(nshape)

        offset = 0
        for patch in self.patches:
            N = patch.N[0]
            self.cellCentreMatrix[offset:offset+N, :] = patch.cellCenters()
            offset += N

