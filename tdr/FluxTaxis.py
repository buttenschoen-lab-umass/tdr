#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np
from numba import jit

from tdr.helpers import VanLeer, offdiagonal
from tdr.Flux import Flux


class TaxisFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix,
                 transitionMatrixAdhesion, *args, **kwargs):
        # taxis call
        self._taxisCall = None
        self._adhCall   = None
        self._finish    = None
        self._bc_call   = None

        # special for the taxis flux
        self.transAdh   = transitionMatrixAdhesion

        # Call parent constructor
        super(TaxisFlux, self).__init__(noPDEs, dimensions, transitionMatrix, *args, **kwargs)

        # extra vars for Taxis Flux
        self.velNonZero = False
        self.limiter    = VanLeer

        # set priority
        self._priority   = 20


    """ Name """
    def __str__(self):
        return 'Taxis'


    """ Setup """
    def _setup(self):
        if self.dim == 1:
            self._finish    = self._finish_1d

            if np.issubdtype(self.trans.dtype, np.number):
                # and check whether we really have non-zero off-diagonal elements
                if np.any(offdiagonal(self.trans)) != 0:
                    print('Constant taxis coefficient!')
                    self._taxisCall = self._flux_1d_const
                else:
                    self._taxisCall = self._dummy_flux
            else:
                print('Variable taxis coefficient!')
                self._taxisCall = self._flux_1d_variable

            # check for variable adhesion coefficient matrix
            if np.issubdtype(self.transAdh.dtype, np.number):
                print('Constant adhesion coefficient!')
                self._adhCall = self._adh_flux_1d
            else:
                print('Variable adhesion coefficients!')
                self._adhCall = self._adh_flux_nonlinear_1d

            # TODO: FIXME doesn't work for more than a single patch
            if self.bd.isNoFlux():
                self._bc_call = self._noflux_bc_1d
            else: # Everything else but NoFlux Boundary conditions
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
    def _simple_call(self, patch, t):
        # make sure patch velocity is reset!
        self._reset_velocity(patch.patchId)

        # compute the flux for each of the PDEs
        for i in range(self.n):
            self.velNonZero = False

            for j in range(self.n):
                self._taxisCall(i, j, patch)
                self._adhCall(i, j, patch)

            # only call expensive taxis flux computation if we have non-zero taxis velocity.
            if self.velNonZero:
                self._finish(i, patch)


    """ Initialization function """
    def init(self, patch):
        self._init_velocity(patch)


    """ Implementation details """
    def _init_velocity(self, patch):
        patchId = patch.patchId
        if self.dim == 1:
            # make sure dictionary exists!
            self._attach_object(dict, 'vij')
            self.vij[patchId] = np.zeros(1 + patch.size())

            # set reset function
            self._reset_velocity = self._reset_velocity_1d
        elif self.dim == 2:
            shape = patch.get_shape()

            # make sure dictionaries exist!
            self._attach_object(dict, 'vxij')
            self._attach_object(dict, 'vyij')

            self.vxij[patchId] = np.zeros(shape + [1, 0])
            self.vyij[patchId] = np.zeros(shape + [0, 1])

            # set reset function
            self._reset_velocity = self._reset_velocity_2d
        else:
            assert False, 'not implemented for higher dimensions!'


    """ Fill temporaries with zeros """
    def _reset_velocity_1d(self, patchId):
        self.vij[patchId].fill(0)


    def _reset_velocity_2d(self, patchId):
        self.vxij[patchId].fill(0)
        self.vyij[patchId].fill(0)


    """ 1D - flux approximation for constant coefficients """
    def _flux_1d_const(self, i, j, patch):
        if i == j: # this means diffusion!
            return

        pij   = self.trans[i, j]
        vij   = self.vij[patch.patchId]

        if pij != 0.:
            self.velNonZero = True
            uDx = patch.data.uDx[j, :]
            vij += pij * uDx


    """ 1D - flux approximation for variable coefficients

        Here it is assumed that the coefficient is only a function of quantity
        from which the gradient is computed i.e.

            p(c) grad(c)

    """
    def _flux_1d_variable(self, i, j, patch):
        if i == j: # this means diffusion!
            return

        # TODO suppose that the coefficient is constant for the moment
        pij   = self.trans[i, j]
        vij   = self.vij[patch.patchId]

        if pij != 0.:
            self.velNonZero = True

            # get the state interpolate
            uDx  = patch.data.uDx[j, :]

            # get the average to interpolate value on the cell boundary
            uAvx = patch.data.uAvx[j, :]

            # compute the velocities
            vij += pij(uAvx) * uDx


    """ 2D - flux approximation for constant coefficients """
    def _flux_2d(self, i, j, patch):
        if i == j: # this means diffusion!
            return

        # TODO suppose that the coefficient is constant for the moment
        pij   = self.trans[i, j]
        vxij  = self.vyij[patch.patchId]
        vyij  = self.vxij[patch.patchId]

        if pij != 0.:
            self.velNonZero = True
            uDx = patch.data.uDx[j, :]
            uDy = patch.data.uDy[j, :]
            vxij += pij * uDx
            vyij += pij * uDy


    """ adhesion flux - linear case! """
    def _adh_flux_1d(self, i, j, patch):
        # check if we have an adhesion term
        aij   = self.transAdh[i, j]
        vij   = self.vij[patch.patchId]

        if aij != 0.:
            self.velNonZero = True
            bw = patch.boundaryWidth
            G  = patch.data.y[j, bw:-bw]
            a  = patch.nonLocalGradient(G).real

            vij[-1] += aij * a[-1]
            vij[1:] += aij * a


    """ adhesion flux """
    def _adh_flux_nonlinear_1d(self, i, j, patch):
        # check if we have an adhesion term
        aij   = self.transAdh[i, j]
        vij   = self.vij[patch.patchId]

        if aij is not None and aij != 0:
            self.velNonZero = True
            bw = patch.boundaryWidth

            G  = aij(patch.data.uAvx[:, 1:]) * patch.data.y[j, bw:-bw]
            a  = patch.nonLocalGradient(G).real

            vij[-1] += a[-1]
            vij[1:] += a


    """ dummy flux """
    def _dummy_flux(self, i, j, patch):
        pass


    """
        This function uses the velocity approximations to compute the taxis
        approximations and finally compute HT.

        For the mathematical details see A. Gerisch 2001.
    """

    def _finish_1d(self, i, patch):
        bw    = patch.boundaryWidth
        y     = patch.data.y[i, :]
        vij   = self.vij[patch.patchId]

        #@jit(nopython=True, nogil=True)
        def _finish_detail_1d(y, bw, vij, limiter):
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
            taxisApprox = (ui + 0.5 * limiter(r) * bwd) * (vij>0) * vij

            ## negative velocity
            ## compute smoothness monitor
            r = fwd / (fwd2 - (np.abs(fwd2) < 1.e-14))

            # compute approximation for negative velocities
            taxisApprox += (ui_p1 - 0.5 * limiter(r) * fwd2) * (vij<0) * vij

            return taxisApprox

        taxisApprox = _finish_detail_1d(y, bw, vij, VanLeer)

        # Add any boundary modifications that are required
        self._bc_call(i, patch, taxisApprox)

        # Now compute HT
        patch.data.ydot[i, :] -= (1. / patch.step_size()) * \
                (taxisApprox[1:] - taxisApprox[:-1])


    """ Add any modifications required by boundary conditions """
    def _dummy_bc_1d(self, i, patch, taxisApprox):
        pass


    """ Modifications required for taxis flux in the presence of no-flux BC.
        For this we set:

            Τ(N, i^B) = v_{avg} α_D,

        where

            α_D = max[0, U(i) - 0.5[ U(i - ν ej) - U(i) ],

        and v_{avg} is the average taxis velocity on the domain's boundary.

    """
    def _noflux_bc_1d(self, i, patch, taxisApprox):
        # TODO: Deal with zero diffusion constants etc. and non-zero flux boundaries
        vij = self.vij[patch.patchId]
        taxisApprox[[0, -1]] = patch.data.get_bd_taxis(i, vij[[0, -1]])


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


    """ EXPERIMENTAL: update adh-trans for bifurcation continuations """
    def update_adhtrans(self, trans):
        self.transAdh = trans



if __name__ == '__main__':
    from testing import create_test
    ctx = create_test()

    trans = np.array([[0,1],[1,0]])
    flux = TaxisFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)
