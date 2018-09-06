#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

from tdr.Flux import Flux

import numpy as np


class DiffusionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix, *args, **kwargs):
        super(DiffusionFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # set priority
        self._priority  = 10


    """ internal call """
    def __call__(self, patch):
        for i in range(self.n):
            self.call(i, patch)


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
    def _flux_1d_periodic(self, i, patch):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]
        Hd    = (pii / patch.step_size()) * (uDx[1:] - uDx[:-1])

        # set ydot in data
        patch.data.ydot[i, :] += Hd


    def _flux_1d_noflux(self, i, patch):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]

        # get bd corrections
        uDx[[0, -1]] = patch.data.get_bd_diffusion(i)

        # call ghost point updater
        coefficient = (patch.step_size() / pii) * np.array([-1, 1])
        patch.data.update_ghost_points_noflux(i, coefficient)

        # compute Hd
        Hd      = (pii / patch.step_size()) * (uDx[1:] - uDx[:-1])

        # set ydot in data
        patch.data.ydot[i, :] += Hd


    """ 2D - Implementation """
    def _flux_2d_periodic(self, i, patch):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]
        uDy   = patch.data.uDy[i, :]
        xdiff = (pii / patch.step_size()[0]) * (uDx[1:, :] - uDx[:-1, :])
        ydiff = (pii / patch.step_size()[1]) * (uDy[:, 1:] - uDy[:, :-1])

        # set ydot in data
        patch.data.ydot[i, :, :] += xdiff + ydiff

