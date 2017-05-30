#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
"""
    This class implements the scratch space for a given patch. It also compute
    several gradients, which are later used by the flux computations.
"""

import numpy as np

class Data(object):
    """
        Scratch space for computations of ydot in a patch.

        Parameters
        ----------
        n               : int
                          Number of PDEs
        patchId         : int
                          Id of the corresponding patch
        dX              : ndarray
                          Array of step lengths in each dimension
        boundaryWidth   : int
                          Width of points to add for boundary conditions
        dimensions      : int
                          Spatial dimensions


        Attributes
        ----------

        TODO

    """
    def __init__(self, n, patchId, dX, boundaryWidth, dimensions, ngb=None):
        # Important numbers
        self.n                  = n
        self.dim                = dimensions
        self.patchId            = patchId
        self.boundaryWidth      = boundaryWidth
        self.dX                 = dX
        self.ngb                = ngb
        # data
        self.ydot               = None
        # storage for y
        self.y                  = None
        self.t                  = None
        # data which is created by this class
        self._reset()


        # call to set face data
        self._compute = None

        # setup
        self._setup()


    """ setup function """
    def _setup(self):
        if self.dim == 1:
            self._compute = self._compute_face_data_1d
        else:
            assert False, 'not implemented'


    """ reset """
    def _reset(self):
        self.ComputedFaceData   = False
        # derivatives on cell boundary
        self.uDx                = None
        self.uDy                = None
        # averages on cell boundary
        self.uAvx               = None
        self.uAvx               = None
        self.skalYT             = None
        self.skalYB             = None


    """ Set values """
    def set_values(self, t, y):
        self.t              = t
        bw                  = self.boundaryWidth
        nx                  = y.shape[1]
        self.y              = np.empty((self.n, nx + 2 * bw))
        self.y[:]           = np.NaN
        if bw > 0:
            self.y[:, bw:-bw]   = y
        else:
            self.y[:] = y

        # reset ydot
        self.ydot           = np.zeros((self.n, nx))

        # TODO look these up
        # deal with the boundaries 1D only atm
        ngbPatchId = self.ngb['left']

        nCb = np.arange(-bw, 0)
        if ngbPatchId == 1: # periodic
            self.y[:, 0:bw] = self.y[:, nCb + nx + bw]
        else:
            assert False, 'not implemented'

        # TODO look these up
        ngbPatchId = self.ngb['right']

        Cb = np.arange(0, bw)
        if ngbPatchId == 1: # periodic BC
            self.y[:, Cb + nx + bw] = self.y[:, bw + Cb]
        else:
            assert False, 'Not implemented'

        # Finally reset
        self._reset()


    """ Compute Face Data """
    def compute_face_data(self):
        self._compute()


    """ Function compute uDx, uDy, uAvx, uAvy, skalYT, skalYB """
    def _compute_face_data_1d(self):
        self._compute_uDx()
        self._compute_uAvx()
        self.ComputedFaceData = True


    def _compute_uDx(self):
        bw = self.boundaryWidth
        self.uDx  = (1. / self.dX[0]) * \
                (self.y[:, bw:-bw+1] - self.y[:, bw-1:-bw])


    def _compute_uAvx(self):
        bw = self.boundaryWidth
        self.uAvx = 0.5 * (self.y[:, bw:-bw+1] + self.y[:, bw-1:-bw])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # create data object
    data = Data(2, 1)

    L = 10
    n = L * 2**5
    h = 1. / (2**5)
    x = np.linspace(0, L, n)
    y1 = np.sin(2. * np.pi * x / L)
    e1 = (2. * np.pi / L) * np.cos(2. * np.pi * x / L)
    y2 = np.cos(2. * np.pi * x / L)
    e2 = -(2. * np.pi / L) * np.sin(2. * np.pi * x / L)

    data.boundaryWidth = 2
    data.set_values(np.vstack((y1, y2)))
    data.h = h
    data._compute_uDx()
    data._compute_uAvx()

    plt.figure(figsize=(15, 7.5))
    plt.subplot(1,3,1)
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(1,3,2)
    plt.plot(x, data.uDx[0, 1:-1], label='d/dx y1')
    plt.plot(x, e1, label='e1')
    plt.legend(loc='best')
    plt.grid()

    plt.subplot(1,3,3)
    plt.plot(x, data.uDx[1, 1:-1], label='d/dx y2')
    plt.plot(x, e2, label='e2')
    plt.legend(loc='best')
    plt.grid()

    plt.show()











