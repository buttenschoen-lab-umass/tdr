#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np

from tdr.utils import VanLeer
from tdr.Flux import Flux


class TaxisFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix,
                 transitionMatrixAdhesion, *args, **kwargs):
        # taxis call
        self._taxisCall = None
        self._adhCall   = None
        self._finish    = None
        self._bc_call   = None

        # Call parent constructor
        super(TaxisFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # extra vars for Taxis Flux
        self.velNonZero = False
        self.limiter    = VanLeer
        self.transAdh   = transitionMatrixAdhesion

        # set priority
        self._priority   = 20


    """ Setup """
    def _setup(self):
        if self.dim == 1:
            self._taxisCall = self._flux_1d
            self._adhCall   = self._adh_flux_1d
            self._finish    = self._finish_1d

            if self.bd.isNoFlux():
                self._bc_call = self._noflux_bc_1d
            else:
                self._bc_call = self._dummy_bc_1d

        elif self.dim == 2:
            self._taxisCall = self._flux_2d
            self._adhCall   = self._dummy_flux
            self._finish    = self._finish_2d
        else:
            assert False, 'At the moment we only support 1D and 2D simulations.'

        # TODO: improve get rid of all the todos in simple_call
        self.call = self._simple_call


    """ Call function """
    def _simple_call(self, patch):
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

            # TODO!
            #self.adh_mult = uu * self.y[i, bw:-bw]

            for j in range(self.n):
                self._taxisCall(i, j, patch)
                self._adhCall(i, j, patch)

            if self.velNonZero:
                self._finish(i, patch)


    """ Implementation details """
    def _init_vel(self, patch):
        if self.dim == 1:
            self.vij = np.zeros(1 + patch.size())
        elif self.dim == 2:
            shape = patch.get_shape()
            self.vxij = np.zeros(shape + [1, 0])
            self.vyij = np.zeros(shape + [0, 1])
        else:
            assert False, 'not implemented for higher dimensions!'


    def _flux_1d(self, i, j, patch):
        if i == j: # this means diffusion!
            return

        # TODO suppose that the coefficient is constant for the moment
        pij   = self.trans[i, j]
        #print('p[%d, %d]: %.2f:' % (i,j,pij))
        if pij != 0.:
            self.velNonZero = True
            uDx = patch.data.uDx[j, :]
            self.vij += pij * uDx


    def _flux_2d(self, i, j, patch):
        if i == j: # this means diffusion!
            return

        # TODO suppose that the coefficient is constant for the moment
        pij   = self.trans[i, j]
        #print('p[%d, %d]: %.2f:' % (i,j,pij))
        if pij != 0.:
            self.velNonZero = True
            uDx = patch.data.uDx[j, :]
            uDy = patch.data.uDy[j, :]
            self.vxij += pij * uDx
            self.vyij += pij * uDy


    """ adhesion flux """
    def _adh_flux_1d(self, i, j, patch):
        # check if we have an adhesion term
        # TODO only implements depends u atm + linear function
        aij   = self.transAdh[i, j]
        if aij != 0.:
            self.velNonZero = True
            bw = patch.boundaryWidth
            G  = patch.data.y[j, bw:-bw] * self.adh_mult
            a  = patch.nonLocalGradient(G).real
            A  = np.hstack((a[-1], a))

            if np.isnan(np.sum(A)):
                raise ValueError("Encountered NaN in non-local gradient calculation")

            self.vij += aij * A


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


    """ 2D implementation """
    def _finish_2d(self, i, patch):
        bw = patch.boundaryWidth
        y  = patch.data.y[i, :]
        dx = patch.step_size()

        # Compute the velocity for the x-coordinate
        # get the current state, and the state shifted by one see (3.12)
        ui          = y[bw-1:-bw, bw:-bw]
        ui_p1       = y[bw:-bw+1, bw:-bw]

        # compute differences between cells
        fwd  = ui_p1                - ui
        bwd  = ui                   - y[:-bw-1, bw:-bw]
        fwd2 = y[bw+1:, bw:-bw]     - ui_p1

        # Positive velocity
        # compute smoothness monitor
        r = fwd / (bwd - (np.abs(bwd) < 1.e-14))

        # approximation for positive velocities
        taxisApprox = (ui + 0.5 * self.limiter(r) * bwd) * (self.vxij>0) * self.vxij

        ## negative velocity
        ## compute smoothness monitor
        r = fwd / (fwd2 - (np.abs(fwd2) < 1.e-14))

        # compute approximation for negative velocities
        taxisApprox += (ui_p1 - 0.5 * self.limiter(r) * fwd2) * (self.vxij<0) * self.vxij

        # Now compute HT
        patch.data.ydot[i, :] -= (1. / dx[0]) * (taxisApprox[1:, :] - taxisApprox[:-1, :])

        # Compute the velocity for the y-coordinate
        # get the current state, and the state shifted by one see (3.12)
        ui          = y[bw:-bw, bw-1:-bw]
        ui_p1       = y[bw:-bw, bw:-bw+1]

        # compute differences between cells
        fwd  = ui_p1                - ui
        bwd  = ui                   - y[bw:-bw, :-bw-1]
        fwd2 = y[bw:-bw, bw+1:]     - ui_p1

        # Positive velocity
        # compute smoothness monitor
        r = fwd / (bwd - (np.abs(bwd) < 1.e-14))

        # approximation for positive velocities
        taxisApprox = (ui + 0.5 * self.limiter(r) * bwd) * (self.vyij>0) * self.vyij

        ## negative velocity
        ## compute smoothness monitor
        r = fwd / (fwd2 - (np.abs(fwd2) < 1.e-14))

        # compute approximation for negative velocities
        taxisApprox += (ui_p1 - 0.5 * self.limiter(r) * fwd2) * (self.vyij<0) * self.vyij

        # Now compute HT
        patch.data.ydot[i, :] -= (1. / dx[1]) * (taxisApprox[:, 1:] - taxisApprox[:, :-1])


if __name__ == '__main__':
    from testing import create_test
    ctx = create_test()

    trans = np.array([[0,1],[1,0]])
    flux = TaxisFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)



