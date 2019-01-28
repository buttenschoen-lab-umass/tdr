#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np

from tdr.utils import LimiterFactory
from tdr.Flux import Flux


class AdvectionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix, *args, **kwargs):
        # taxis call
        self._advCall   = None
        self._finish    = None
        self._bc_call   = None

        # Call parent constructor
        super(AdvectionFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # extra vars for Taxis Flux
        self.velNonZero = False
        self.limiter    = LimiterFactory(kwargs.pop('limiter', 'vanLeer'))

        # set priority
        self._priority   = 20


    """ Name """
    def __str__(self):
        return 'Advection'


    """ Setup """
    def _setup(self):
        if self.dim == 1:
            self._advCall = self._flux_1d
            self._finish    = self._finish_1d

            # we can use the same function as Data takes care of all the
            # important logic!
            if self.bd.isNoFlux() or self.bd.isDirichlet():
                self._bc_call = self._noflux_bc_1d
            else:
                self._bc_call = self._dummy_bc_1d

        else:
            assert False, 'At the moment we only support 1D simulations with advection.'

        # TODO: improve get rid of all the todos in simple_call
        self.call = self._simple_call


    """ Call function """
    def _simple_call(self, patch, t):
        # compute the flux for each of the PDEs
        # TODO: test
        bw = patch.boundaryWidth
        uu = np.ones_like(patch.data.y[:, bw:-bw])
        #for i in range(self.n):
        #    uu -= patch.data.y[i, bw:-bw]
        #m = np.where(uu < 0.)
        #uu[m] = 0.
        self.adh_mult = uu

        for i in range(self.n):
            # FIXME works only in 1D atm!
            self.velNonZero = False

            self._init_vel(patch)

            for j in range(self.n):
                self._advCall(i, j, patch)

            if self.velNonZero:
                self._finish(i, patch)


    """ Implementation details """
    def _init_vel(self, patch):
        if self.dim == 1:
            self.vij = np.zeros(1 + patch.size())
        else:
            assert False, 'not implemented for higher dimensions!'


    def _flux_1d(self, i, j, patch):
        pij   = self.trans[i, j]
        if pij != 0.:
            self.velNonZero = True
            # Make this dependable on u and t
            self.vij += pij


    """ dummy flux """
    def _dummy_flux(self, i, j, patch):
        pass


    """
        This function uses the velocity approximations to compute the taxis
        approximations and finally compute HT.

        For the mathematical details see A. Gerisch 2001.
    """
    def _finish_1d(self, i, patch):
        bw = patch.boundaryWidth
        y  = patch.data.y[i, :]

        # get the current state, and the state shifted by one see (3.12)
        ui      = y[bw-1:-bw]
        ui_p1   = y[bw:-bw+1]

        # compute differences between cells
        fwd  = ui_p1        - ui
        bwd  = ui           - y[:-bw-1]
        fwd2 = y[bw+1:]     - ui_p1

        # Positive velocity
        # compute smoothness monitor
        r = fwd / (bwd - (np.abs(bwd) < 1.e-14))

        # approximation for positive velocities
        taxisApprox = (ui + 0.5 * self.limiter(r) * bwd) * (self.vij>0) * self.vij

        ## negative velocity
        ## compute smoothness monitor
        r = fwd / (fwd2 - (np.abs(fwd2) < 1.e-14))

        # compute approximation for negative velocities
        taxisApprox += (ui_p1 - 0.5 * self.limiter(r) * fwd2) * (self.vij<0) * self.vij

        # Add any boundary modifications that are required
        self._bc_call(i, patch, taxisApprox)

        # Now compute HT
        patch.data.ydot[i, :] -= (1. / patch.step_size()) * \
                (taxisApprox[1:] - taxisApprox[:-1])


    """ Add any modifications required by boundary conditions """
    def _dummy_bc_1d(self, i, patch, taxisApprox):
        pass


    """ Modifications required for no-flux bc """
    def _noflux_bc_1d(self, i, patch, taxisApprox):
        # TODO: Deal with zero diffusion constants etc.
        taxisApprox[[0, -1]] = patch.data.get_bd_taxis(i, self.vij[[0, -1]])
        return

