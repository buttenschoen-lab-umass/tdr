#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from Flux import Flux


class DiffusionFlux(Flux):
    def __init__(self, noPDEs, dimensions, transitionMatrix):
        super(DiffusionFlux, self).__init__(noPDEs, dimensions, transitionMatrix)

        # set priority
        self.priority = 10


    """ i: is the number of PDE """
    def __call__(self, patchData):
        for i in range(self.n):
            # Dont seem to needs this any longer!
            self._flux_1d_periodic(i, patchData)


    def _switchBoard(self, i, patchData):
        # TODO optimize?
        bd = patchData.ngb
        if bd.isPeriodic():
            self._flux_1d_periodic(i, patchData)
        elif bd.isNeumann():
            self._flux_1d_neumann(i, patchData)
        elif bd.isDirichlet():
            self._flux_1d_dirichlet(i, patchData)
        else:
            assert False, 'Unknown boundary condition!'


    """ Computational details: The functions compute H_D(U, i). """
    def _flux_1d_periodic(self, i, patch):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]
        Hd    = (pii / patch.step_size()) * (uDx[1:] - uDx[:-1])

        # set ydot in data
        patch.data.ydot[i, :] += Hd


    def _flux_1d_neumann(self, i, patch):
        pii   = self.trans[i, i]
        uDx   = patch.data.uDx[i, :]

        #print('uDx:', uDx)

        # Since we have neumann boundary conditions set the left and right
        # point of uDx
        #lB      = patch.ngb.left
        #rB      = patch.ngb.right
        #uDx[1]  = - lB.oNormal * lB.value
        #uDx[-1] = - rB.oNormal * rB.value

        #print('after: uDx:', uDx)

        Hd      = (pii / patch.step_size()) * (uDx[1:] - uDx[:-1])

        # set ydot in data
        patch.data.ydot[i, :] += Hd

