#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function

import numpy as np

from utils import VanLeer
from Flux import Flux


class TaxisFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix,
                 transitionMatrixAdhesion, *args, **kwargs):
        # Call parent constructor
        super(TaxisFlux, self).__init__(noPDEs, dimensions, transitionMatrix)

        # extra vars for Taxis Flux
        self.velNonZero = False
        self.limiter    = VanLeer
        self.transAdh   = transitionMatrixAdhesion

        # set priority
        self.priority   = 20

        # taxis call
        self._taxisCall = None


    """ Setup """
    #def _setup(self):
    #    nonLocalTaxis = np.any(self.transAdh != 0)
    #    localTaxis    = np.any(self.trans != 0)

    #    if nonLocalTaxis and localTaxis:
    #        assert False, 'Can\'t be both non-local and local!'

    #    if nonLocalTaxis:
    #        self._taxisCall = self._adh_flux_1d
    #    elif localTaxis:
    #        self._taxisCall = self._flux_1d
    #    else:
    #        print("Using a dummy flux!")
    #        self._taxisCall = self._dummy_flux


    """ i: is the number of PDE """
    def __call__(self, patch):
        # compute the flux for each of the PDEs
        for i in range(self.n):
            # FIXME works only in 1D atm!
            self.velNonZero = False

            self.vij = np.zeros(1 + patch.size())
            #print('vij=',self.vij.shape)

            for j in range(self.n):
                # self.call(i, j, patch)
                self._flux_1d(i, j, patch)
                self._adh_flux_1d(i, j, patch)

            if self.velNonZero:
                self._finish(i, patch)


    """ Implementation details """
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


    """ adhesion flux """
    def _adh_flux_1d(self, i, j, patch):
        # check if we have an adhesion term
        # TODO only implements depends u atm + linear function
        aij   = self.transAdh[i, j]
        if aij != 0.:
            self.velNonZero = True
            bw = patch.boundaryWidth
            G  = patch.data.y[j, bw:-bw]
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
    def _finish(self, i, patch):
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

        # Now compute HT
        patch.data.ydot[i, :] -= (1. / patch.step_size()) * \
                (taxisApprox[1:] - taxisApprox[:-1])

        # debug
        #
        # now that H_T + div(u(t, x) p grad(c)) = O(h^2)
        #
        #HT = (-1. / patch.step_size()) * (taxisApprox[1:] - taxisApprox[:-1])
        #flux = (1. / patch.step_size()) * self.vij * y[bw-1:-bw]
        #dflux = flux[1:] - flux[:-1]
        #error = HT + dflux
        #print('Error:', np.max(error))
        #print('\tE:',error[0:10])

        #patch.data.ydot[i,0] = 1.05 * patch.data.ydot[i,1]

        #print('\t\ttaxisLeft:', taxisApprox[:5])
        #print('\t\tleft:',patch.data.ydot[i, :5])
        #print('\t\tVlef:',self.vij[:5])
        #print('\t\ttaxisRigh:', taxisApprox[-5:])
        #print('\t\tright:',patch.data.ydot[i, -5:])



if __name__ == '__main__':
    from testing import create_test
    ctx = create_test()

    trans = np.array([[0,1],[1,0]])
    flux = TaxisFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)



