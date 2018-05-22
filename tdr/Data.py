#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function

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

        Data Layout
        ----------
        Indexing works like this y[pdeIdx, xCoord, yCoord]
        Note that if we print a 2D numpy array it looks like this

        (x0, y0) ----------------------------------------- (x0, y1)
           |                                                  |
           |                                                  |
           |                                                  |
           |                                                  |
           |                                                  |
           |                                                  |
           |                                                  |
           |                                                  |
        (x1, y0) ----------------------------------------- (x1, y1)

        The access indices looks like this:

        (0, 0) ----------------------------------------- (0, Ny)
          |                                                |
          |                                                |
          |                                                |
          |                                                |
          |                                                |
          |                                                |
          |                                                |
          |                                                |
        (Nx, 0) --------------------------------------- (Nx, Ny)


        Attributes
        ----------

        TODO

    """
    def __init__(self, n, patchId, dX, boundaryWidth, dimensions, ngb):
        # Important numbers
        self.n                  = n
        self.dim                = dimensions
        self.patchId            = patchId
        self.boundaryWidth      = boundaryWidth
        self.dX                 = dX
        self.boundary           = ngb
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
        #assert self.boundary is not None, 'Boundary cannot be None!'
        print('Data Dim:',self.dim)
        if self.dim == 1:
            self._compute = self._compute_face_data_1d
        elif self.dim == 2:
            self._compute = self._compute_face_data_2d
        else:
            assert False, 'Compute face data not implemented for %dd' % self.dim


    """ reset """
    def _reset(self):
        self.ComputedFaceData   = False
        # derivatives on cell boundary
        self.uDx                = None
        self.uDy                = None
        # averages on cell boundary
        self.uAvx               = None
        self.uAvy               = None
        self.skalYT             = None
        self.skalYB             = None

        if self.dim == 1:
            self.set_values = self.set_values_1d
        elif self.dim == 2:
            self.set_values = self.set_values_2d
        else:
            assert False, 'Compute face data not implemented for %dd' % self.dim


    """ Set values """
    def set_values_1d(self, t, y):
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
        lBoundary = self.boundary.left

        if lBoundary.isPeriodic(): # periodic
            # TODO do this better!
            nCb = np.arange(-bw, 0)
            self.y[:, 0:bw] = self.y[:, nCb + nx + bw]

        # at the moment this is really NoFlux bc
        # TODO deal with this per equation!
        elif lBoundary.isNeumann():
            # This setups the boundary as follows: Here || denotes the boundary
            # | y1 | y0 || y0 | y1 |
            nCb = np.arange(bw, 0, -1) - 1
            self.y[:, 0:bw] = self.y[:, nCb + bw]
            #print('NeumannLeft:',self.y[1, :5])
        else:
            assert False, 'not implemented'

        # TODO look these up
        rBoundary = self.boundary.right

        if rBoundary.isPeriodic(): # periodic BC
            Cb = np.arange(0, bw)
            self.y[:, Cb + nx + bw] = self.y[:, bw + Cb]
        # NoFlux boundary conditions only here!!!
        elif rBoundary.isNeumann():
            # This setups the boundary as follows: Here || denotes the boundary
            # | yN-1 | yN || yN | yN-1 |
            Cb  = np.arange(0, bw)
            nCb = np.arange(1, -bw+1, -1)
            self.y[:, Cb + nx + bw] = self.y[:, nCb + nx]
            #print('NeumannRight:',self.y[1, -5:])
        else:
            assert False, 'Not implemented'

        # Finally reset
        self._reset()


    """ Set values """
    def set_values_2d(self, t, y):
        self.t              = t
        bw                  = self.boundaryWidth
        nx                  = y.shape[1]
        ny                  = y.shape[2]
        self.y              = np.empty((self.n, nx + 2 * bw, ny + 2 * bw))
        self.y[:]           = np.NaN
        if bw > 0:
            self.y[:, bw:-bw, bw:-bw]   = y
        else:
            self.y[:] = y

        # reset ydot
        self.ydot           = np.zeros((self.n, nx, ny))

        # let's define periodic boundary conditions by default for the begining
        # x-boundary
        nCb = np.arange(-bw, 0)
        self.y[:, 0:bw, bw:-bw] = self.y[:, nCb + nx + bw, bw:-bw]
        self.y[:, bw:-bw, 0:bw] = self.y[:, bw:-bw, nCb + nx + bw]

        Cb = np.arange(0, bw)
        self.y[:, Cb + nx + bw, bw:-bw] = self.y[:, bw + Cb, bw:-bw]
        self.y[:, bw:-bw, Cb + nx + bw] = self.y[:, bw:-bw, bw + Cb]

        # Finally reset
        self._reset()


    """ Grow the domain """
    def elongate(self, where, direction, h = 0.1):
        pass


    """ Compute Face Data """
    def compute_face_data(self):
        self._compute()


    """ Function compute uDx, uAvx, skalYT, skalYB """
    def _compute_face_data_1d(self):
        self._compute_uDx_1d()
        self._compute_uAvx_1d()
        self.ComputedFaceData = True


    """ Function compute uDx, uDy, uAvx, uAvy, skalYT, skalYB """
    def _compute_face_data_2d(self):
        self._compute_uDx_2d()
        self._compute_uDy_2d()
        self._compute_uAvx_2d()
        self._compute_uAvy_2d()
        self.ComputedFaceData = True


    """ This computes uDx: the x-derivative approximation on right cell boundaries """
    def _compute_uDx_1d(self):
        bw = self.boundaryWidth
        assert bw == 2, 'BoundaryWidth must be at least 2 for fluxes!'

        self.uDx  = (1. / self.dX[0]) * \
                (self.y[:, bw:-bw+1] - self.y[:, bw-1:-bw])


    def _compute_uAvx_1d(self):
        bw = self.boundaryWidth
        self.uAvx = 0.5 * (self.y[:, bw:-bw+1] + self.y[:, bw-1:-bw])


    """ Functions that compute required data in 2D """
    def _compute_uDx_2d(self):
        bw = self.boundaryWidth
        assert bw == 2, 'BoundaryWidth must be at least 2 for fluxes!'

        self.uDx  = (1. / self.dX[0]) * \
                (self.y[:, bw:-bw+1, bw:-bw] - self.y[:, bw-1:-bw, bw:-bw])


    def _compute_uDy_2d(self):
        bw = self.boundaryWidth
        assert bw == 2, 'BoundaryWidth must be at least 2 for fluxes!'

        self.uDy  = (1. / self.dX[1]) * \
                (self.y[:, bw:-bw, bw:-bw+1] - self.y[:, bw:-bw, bw-1:-bw])


    def _compute_uAvx_2d(self):
        bw = self.boundaryWidth
        self.uAvx = 0.5 * (self.y[:, bw:-bw+1, bw:-bw] + self.y[:, bw-1:-bw, bw:-bw])


    def _compute_uAvy_2d(self):
        bw = self.boundaryWidth
        self.uAvy = 0.5 * (self.y[:, bw:-bw, bw:-bw+1] + self.y[:, bw:-bw, bw-1:-bw])


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











