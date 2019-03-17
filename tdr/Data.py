#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

"""
    This class implements the scratch space for a given patch. It also compute
    several gradients, which are later used by the flux computations.
"""

import numpy as np

from tdr.utils import asarray


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

        1D Data Layout
        ----------
        Data is in a 1d numpy array from the left to the right.

           Ω1      Ω2              ΩM
        |       |      |        |      |
        |  x1   |  x2  | ...... |  xM  |
        |       |      |        |      |
        Γ                              Γ

        where Γ is the domain boundary.

        Some boundary conditions require the value of u on the boundary. For
        this purpose we use a simple extrapolation. So that the value on the
        left boudary is computed by:

            α_{D} = N_{1} + 0.5 (N_{2} - N_{1}),

        and the right boundary by:

            α_{D} = N_{M} - 0.5 (N_{M-1} - N_{M}).

        2D Data Layout
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
    def __init__(self, n, patchId, dX, boundaryWidth, dimensions, ngb, *args, **kwargs):
        # Important numbers
        self.n                  = n
        self.dim                = dimensions
        self.patchId            = patchId
        self.boundaryWidth      = boundaryWidth
        self.dX                 = dX
        self.boundary           = ngb

        # data
        self.ydot               = None

        # patch scaling information for deformation support
        self.r                  = 1.
        self.dr                 = 0.
        self.r_dot_over_r       = 0.

        # storage for y
        self.y                  = None
        self.t                  = None

        # Compute time-step for deformation support
        self.dt                 = None

        # patch deformation
        self.deformation        = None
        self.deformation_dt     = None

        # approximations of values of y on the boundary a.k.a. α_{D}.
        self.boundary_approx    = None

        # needed for no-flux bc
        self.taxis_bd_aprx      = None
        self.diffusion_bd_aprx  = None

        # data which is created by this class
        self._reset()

        # call to set face data
        self._compute           = None

        # helper to deal with boundary condition related updates
        self._boundary_helper   = None

        # setup
        self._setup(*args, **kwargs)


    """ setup function """
    def _setup(self, *args, **kwargs):
        #assert self.boundary is not None, 'Boundary cannot be None!'
        if self.dim == 1:
            self._compute = self._compute_face_data_1d
            self.set_values = self.set_values_1d
        elif self.dim == 2:
            self._compute = self._compute_face_data_2d
            self.set_values = self.set_values_2d
        else:
            assert False, 'Compute face data not implemented for %dd' % self.dim

        # set initial time
        self.t = kwargs.pop('t0', 0.)

        # setup possible domain deformation
        self.setup_deformation(*args, **kwargs)

        # setup boundary conditions
        lBoundary = self.boundary.left
        rBoundary = self.boundary.right

        if lBoundary.isPeriodic() or rBoundary.isPeriodic():
            assert lBoundary.isPeriodic(), 'Error: Both boundaries are required to be periodic!'
            assert rBoundary.isPeriodic(), 'Error: Both boundaries are required to be periodic!'
            self._boundary_helper = self._periodic_set_values_helper
        elif lBoundary.isNeumann() or rBoundary.isNeumann():
            assert lBoundary.isNeumann(), 'Error: Both boundaries are required to be Neumann!'
            assert rBoundary.isNeumann(), 'Error: Both boundaries are required to be Neumann!'
            self._boundary_helper = self._neumann_set_values_helper
        elif lBoundary.isNoFlux() or rBoundary.isNoFlux():
            assert lBoundary.isNoFlux(), 'Error: Both boundaries are required to be Neumann!'
            assert rBoundary.isNoFlux(), 'Error: Both boundaries are required to be Neumann!'
            self._boundary_helper = self._noflux_set_values_helper
        elif lBoundary.isDirichlet() or rBoundary.isDirichlet():
            assert lBoundary.isDirichlet(), 'Error: Both boundaries are required to be Dirichlet!'
            assert rBoundary.isDirichlet(), 'Error: Both boundaries are required to be Dirichlet!'
            self._boundary_helper = self._dirichlet_set_values_helper
        else:
            print("WARNING: No special boundary handling enabled!")


    """ setup deformation """
    def setup_deformation(self, *args, **kwargs):
        # see if we have a deforming domain
        deformation     = asarray(kwargs.pop('r',  None))
        deformation_dt  = asarray(kwargs.pop('dr', None))

        assert deformation.size == 1 and deformation_dt.size == 1, \
            'We can only have one function describing domain deformation!'
        self.deformation    = deformation[0, 0]
        self.deformation_dt = deformation_dt[0, 0]

        # We are not deforming
        if self.deformation is None or self.deformation_dt is None:
            return

        #assert self.deformation(0, 0) == 1., 'Deformation must return 1. for time zero!'
        #self.r = self.deformation(0, 0)


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


    """ update length information """
    def deform(self, t, y):
        self.r                  = self.deformation(t, y)
        self.dr                 = self.deformation_dt(t, y)

        assert self.r > 0., 'Domain scaling factor cannot be non-positive!'

        # can't approximate dr/dt by a finite difference since the rowmap
        # solver will call these out of sequence.
        # self.r_dot_over_r       = (1. - self.r_previous / self.r) / self.dt
        self.r_dot_over_r       = self.dr / self.r


    """ Set values """
    def _set_timestep(self, t):
        self.t              = t


    def set_values_1d(self, t, y):
        # compute time-step
        self._set_timestep(t)

        bw                  = self.boundaryWidth
        nx                  = y.shape[1]
        self.y              = np.empty((self.n, nx + 2 * bw))
        self.y[:]           = np.NaN
        if bw > 0:
            self.y[:, bw:-bw]   = y
        else:
            self.y[:] = y

        # make sure this runs before deform!
        self.r_dot_over_r       = 0.

        # reset ydot
        self.ydot           = np.zeros((self.n, nx))

        # Deal with any boundary updates that are required by the boundary conditions
        self._boundary_helper(bw, nx)

        # do possible deformation
        if self.deformation is not None:
            self.deform(t, y)

        # Finally reset
        self._reset()


    """ Helpers for dealing with boundary conditions """
    def _periodic_set_values_helper(self, bw, nx):
        # Left boundary
        nCb = np.arange(-bw, 0)
        self.y[:, 0:bw] = self.y[:, nCb + nx + bw]

        # Right boundary
        Cb = np.arange(0, bw)
        self.y[:, Cb + nx + bw] = self.y[:, bw + Cb]


    def _neumann_set_values_helper(self, bw, nx):
        # Left Boundary

        # This setups the boundary as follows: Here || denotes the boundary
        # | y1 | y0 || y0 | y1 |
        nCb = np.arange(bw, 0, -1) - 1
        self.y[:, 0:bw] = self.y[:, nCb + bw]

        # Right boundary

        # This setups the boundary as follows: Here || denotes the boundary
        # | yN-1 | yN || yN | yN-1 |
        Cb  = np.arange(0, bw)
        nCb = np.arange(1, -bw+1, -1)
        self.y[:, Cb + nx + bw] = self.y[:, nCb + nx]


    """ Implementation of no-flux boundary conditions """
    def _noflux_set_values_helper(self, bw, nx):
        # Left Boundary

        # This setups the boundary as follows: Here || denotes the boundary
        # | y1 | y0 || y0 | y1 |
        nCb = np.arange(bw, 0, -1) - 1
        self.y[:, 0:bw] = self.y[:, nCb + bw]

        # Right boundary

        # This setups the boundary as follows: Here || denotes the boundary
        # | yN-1 | yN || yN | yN-1 |
        Cb  = np.arange(0, bw)
        nCb = np.arange(1, -bw+1, -1)
        self.y[:, Cb + nx + bw] = self.y[:, nCb + nx]

        # update boundary values
        self._compute_boundary_approx_noflux_1d(bw)


    """ Implementation of dirichlet boundary conditions """
    def _dirichlet_set_values_helper(self, bw, nx):
        # use the above mentioned extrapolations
        y = self.y[:, bw:-bw]

        # Left Boundary

        # This setups the boundary as follows: Here || denotes the boundary
        # | alpha_D | alpha_D || y0 | y1 |
        mask = self.boundary.left.bc_mask()
        nask = self.boundary.left.nbc_mask()

        # in some place we need to reflect for NoBC
        nCb = np.arange(bw, 0, -1) - 1
        self.y[:, 0:bw] = mask * self.boundary.left(y) + nask * self.y[:, nCb + bw]

        # Right Boundary
        # This setups the boundary as follows: Here || denotes the boundary
        # | yN-1 | yN || alpha_D | alpha_D |
        mask = self.boundary.right.bc_mask()
        nask = self.boundary.right.nbc_mask()

        Cb  = np.arange(0, bw)
        nCb = np.arange(1, -bw+1, -1)
        self.y[:, Cb + nx + bw] = mask * self.boundary.right(y) + nask * self.y[:, nCb + nx]

        # setup boundary approximations
        self._compute_boundary_approx_dirichlet_1d(bw)


    """ Compute boundary approximations in the presence of no-flux BC in 1d.

        This function computes α_D in the case of no-flux BC:

            α_D = max[0, U(i) - 0.5[ U(i - ν ej) - U(i) ].

    """
    def _compute_boundary_approx_noflux_1d(self, bw):
        self.boundary_approx    = np.zeros((self.n, 2))
        self.taxis_bd_aprx      = np.zeros((self.n, 2))
        self.diffusion_bd_aprx  = np.zeros((self.n, 2))

        # use the above mentioned extrapolations
        y = self.y[:, bw:-bw]

        # Here we enforce positivity!
        self.boundary_approx[:, 0] = np.maximum(0, y[:, 0]  + 0.5 * (y[:, 1]  - y[:, 0]))
        self.boundary_approx[:, 1] = np.maximum(0, y[:, -1] - 0.5 * (y[:, -2] - y[:, -1]))


    """ compute boundary approximations for dirichlet conditions in 1d """
    # ATTENTION these are currently setup for a system of hyperbolic equations!
    def _compute_boundary_approx_dirichlet_1d(self, bw):
        self.boundary_approx    = np.zeros((self.n, 2))
        self.taxis_bd_aprx      = np.zeros((self.n, 2))
        self.diffusion_bd_aprx  = np.zeros((self.n, 2))

        # use the above mentioned extrapolations
        y = self.y[:, bw:-bw]

        # Here we enforce positivity!
        self.boundary_approx[:, 0] = self.boundary.left(y)
        self.boundary_approx[:, 1] = self.boundary.right(y)


    """ Compute the boundary face data for the taxis approximation """
    def get_bd_taxis(self, i, v):
        """ This is done in Data since we have to store these values! """
        self.taxis_bd_aprx[i, :] = v * self.boundary_approx[i, :]
        return self.taxis_bd_aprx[i, :]


    """ Compute the boundary face diffusion approximation """
    def get_bd_diffusion(self, i):
        """ This is done in Data since we have to store these values! """
        # TODO: Implement non-zero fluxes!
        self.diffusion_bd_aprx[i, :] = self.taxis_bd_aprx[i, :]
        return self.diffusion_bd_aprx[i, :]


    """ Update the ghost points after imposing no-flux bc

        This function sets:

        U(i + ν ej) = max[0, U(i) + ν h / ε D(U, i^B)]

    """
    def update_ghost_points_noflux(self, i, coefficient):
        assert self.boundaryWidth == 2, 'Check implementation!'
        bw = self.boundaryWidth
        dFluxBd = self.get_bd_diffusion(i)
        zz = np.maximum(0., self.y[i, [bw, -bw-1]] + dFluxBd * coefficient)
        self.y[i, [bw-1, -bw]]  = zz

        # Need to update uDx near the boundary!
        # TODO improve this!
        self.uDx[i, :]  = (1. / self.dX[0]) * \
                (self.y[i, bw:-bw+1] - self.y[i, bw-1:-bw])


    """ Set values """
    def set_values_2d(self, t, y):
        # compute time-step
        self._set_timestep(t)

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
        assert bw == 2, 'BoundaryWidth set at %d must be at least 2 for fluxes!' % bw

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


