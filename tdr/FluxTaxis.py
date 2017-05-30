#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

import numpy as np

from utils import VanLeer
from Flux import Flux


class TaxisFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix,
                 transitionMatrixAdhesion):
        super(TaxisFlux, self).__init__(noPDEs, dimensions, transitionMatrix)

        # extra vars for Taxis Flux
        self.velNonZero = False
        self.limiter    = VanLeer
        self.transAdh   = transitionMatrixAdhesion

        # set priority
        self.priority = 20


    """ i: is the number of PDE """
    def __call__(self, patch):
        # compute the flux for each of the PDEs
        for i in range(self.n):
            # FIXME works only in 1D atm!

            self.vij = np.zeros(1 + patch.size())
            #print('vij=',self.vij.shape)

            for j in range(self.n):
                self.call(i, j, patch)
                self._adh_flux_1d(i, j, patch)

            if self.velNonZero:
                self._finish(i, patch)


    """ Implementation details """
    def _flux_1d(self, i, j, patch):
        if i == j: # this has already been done
            return

        # TODO suppose that the coefficient is constant for the moment
        pij   = self.trans[i, j]
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


    """
        This function uses the velocity approximations to compute the taxis
        approximations and finally compute HT.

        For the mathematical details see A. Gerisch 2001.
    """
    def _finish(self, i, patch):
        bw = patch.boundaryWidth
        y  = patch.data.y[i, :]

        # compute differences between cells
        fwd  = y[bw:-bw+1]  - y[bw-1:-bw]
        bwd  = y[bw-1:-bw]  - y[bw-2:-bw-1]
        fwd2 = y[bw+1:]     - y[bw:-bw+1]

        # Positive velocity
        # compute smoothness monitor
        r = fwd / (bwd - (np.abs(bwd) < 1.e-14))

        # approximation
        taxisApprox = (y[bw:-bw+1] + 0.5 * self.limiter(r) * bwd) * \
                (np.abs(self.vij)>0) * self.vij

        # negative velocity
        # compute smoothness monitor
        r = fwd / (fwd2 - (np.abs(fwd2) < 1.e-14))

        taxisApprox += (y[bw:-bw+1] - 0.5 * self.limiter(r) * fwd2) * \
                (np.abs(self.vij)<0) * self.vij

        # Now compute HT
        patch.data.ydot[i, :] += (-1. / patch.step_size()) * \
                (taxisApprox[1:] - taxisApprox[:-1])


if __name__ == '__main__':
    from testing import create_test
    ctx = create_test()

    trans = np.array([[0,1],[1,0]])
    flux = TaxisFlux(ctx, trans)

    for i in range(0, 2):
        flux(i, 0)



