#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

from tdr.Flux import Flux

import numpy as np
from numba import jit


class DiffusionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix, *args, **kwargs):
        super(DiffusionFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # set priority
        self._priority  = 10


    """ Name """
    def __str__(self):
        return 'Diffusion'


    """ internal call """
    def __call__(self, patch, t):
        for i in range(self.n):
            self.call(i, patch, t)


    """ update constants """
    def update(self, t):
        # HACK
        self.trans = self.cFunctional(t, 0)


    """ setup the flux """
    def _setup(self):
        if self.dim == 1:
            if self.bd.isPeriodic():
                self.call = self._flux_1d_periodic
            elif self.bd.isNeumann():
                # This only works for zero flux bc i.e. Neumann
                self.call = self._flux_1d_periodic
            elif self.bd.isNoFlux():
                self.call = self._flux_1d_noflux
            elif self.bd.isDirichlet():
                self.call = self._flux_1d_dirichlet
            else:
                assert False, 'Unknown boundary condition %s!' % self.bd
        elif self.dim == 2:
            self.call = self._flux_2d_periodic
        else:
            assert False, 'Diffusion Flux not implemented for dimension %d.' % self.dim


    """ Computational details: The functions compute H_D(U, i). """
    def _flux_1d_periodic(self, i, patch, t):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]

        #@jit(nopython=True, nogil=True)
        # this really slows things down!
        def _flux_1d_periodic_detail(ydot, pii, h, uDx):
            ydot += (pii / h) * (uDx[1:] - uDx[:-1])

        _flux_1d_periodic_detail(patch.data.ydot[i, :], pii, patch.step_size(), uDx)


    """ Compute the diffusion flux approximation i.e. H_D when imposing no-flux
        boundary conditions.

        For this we set:

            Τ(N, i^B) = v_{avg} α_D,

        for the definition of α_D see _flux_1d_noflux in TaxisFlux, and we set:

            D(N, i^B) = T(U, i^B) - ν α_F

    """
    def _flux_1d_noflux(self, i, patch, t):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]

        # call ghost point updater
        coefficient = (patch.step_size() / pii) * np.array([-1, 1])
        patch.data.update_ghost_points_noflux(i, coefficient)

        # set ydot in data
        patch.data.ydot[i, :] += (pii / patch.step_size()) * (uDx[1:] - uDx[:-1])


    """ 2D - Implementation """
    def _flux_2d_periodic(self, i, patch, t):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]
        uDy   = patch.data.uDy[i, :]
        xdiff = (pii / patch.step_size()[0]) * (uDx[1:, :] - uDx[:-1, :])
        ydiff = (pii / patch.step_size()[1]) * (uDy[:, 1:] - uDy[:, :-1])

        # set ydot in data
        patch.data.ydot[i, :, :] += xdiff + ydiff
