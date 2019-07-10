#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import absolute_import, print_function, division

"""
    This class implements the scratch space for a given patch. It also compute
    several gradients, which are later used by the flux computations.
"""

import numpy as np

from tdr.helpers import asarray


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
    def __init__(self, ngb=None, *args, **kwargs):
        # Important numbers
        self.n                  = np.asarray(kwargs.pop('n'), np.int)
        self.patchId            = kwargs.pop('patchId')
        self.boundaryWidth      = kwargs.pop('boundaryWidth')
        self.dX                 = kwargs.pop('dX')
        self.N                  = np.asarray(kwargs.pop('N'), np.int)
        self.dim                = self.N.size
        self.boundary           = ngb

        # indices of patch data in y - vector
        self.ps                 = kwargs.pop('start')
        self.pe                 = None

        # data - this is a slice of memory owned by Grid
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

        # dimensions
        self.dims = ['x', 'y', 'z']

        # setup
        self._setup(*args, **kwargs)


    def __str__(self):
        return 'Data(%d)' % self.patchId


    def __repr__(self):
        return 'Data(%d)' % self.patchId


    """ The number of small volumes in the patch """
    def size(self):
        return np.prod(self.N)


    """ Grid size """
    def grid_size(self):
        return self.N

    """ Required memory size """
    def domain_shape(self):
        gsize = self.grid_size()
        for d in range(self.dim):
            # these are required for dealing with boundary conditions
            setattr(self, 'n' + self.dims[d], gsize[d])

        # compute shape for y-vector
        yshape    = (int(self.n),)
        ydotshape = (int(self.n),)
        for d in range(self.dim):
            yshape    += (gsize[d] + 2 * self.bw,)
            ydotshape += (gsize[d],)

        return yshape, ydotshape


    """ The size required to store the PDEs on the patch in memory """
    def memory_size(self):
        return self.size() * self.n


    """ setup """
    def _setup_data(self):
        self.bw     = self.boundaryWidth
        self.dshape, self.yshape = self.domain_shape()

        # commonly used values for boundaries
        self.Cb  = np.arange(0, self.bw)
        self.nCb = np.arange(self.bw, 0, -1) - 1
        self.pCb = np.arange(-self.bw, 0)
        self.mCb = np.arange(1, -self.bw+1, -1)

        self.y              = np.empty(self.dshape)
        self.y[:]           = np.NaN


    """ Set ydot """
    def set_ydot(self, ydot):
        self.ydot = ydot.reshape(self.yshape)


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

        # setup boundary helpers
        if self.dim == 1:
            self._setup_bd_1d()
        elif self.dim == 2:
            self._setup_bd_2d()
        else:
            assert False, 'Boundaries for dimension %d not supported!' % self.dim

        # setup data
        self._setup_data()


    """ Boundary Helper 1D """
    def _setup_bd_1d(self):
        # setup boundary conditions
        lBoundary = self.boundary.left
        rBoundary = self.boundary.right

        if lBoundary.isSeparable() or rBoundary.isSeparable():
            assert lBoundary.isSeparable(), 'Error: Both boundaries are required to be separable!'
            assert rBoundary.isSeparable(), 'Error: Both boundaries are required to be separable!'

            if lBoundary.isNeumann():
                self._boundary_helper_left = self._neumann_set_values_helper_left
            elif lBoundary.isGlue():
                self._boundary_helper_left = self._glue_set_values_helper_left
            else:
                assert False, 'Can\'t happen!'

            if rBoundary.isNeumann():
                self._boundary_helper_right = self._neumann_set_values_helper_right
            elif rBoundary.isGlue():
                self._boundary_helper_right = self._glue_set_values_helper_right
            else:
                assert False, 'Can\'t happen!'

            self._boundary_helper = self._set_values_helper
        elif lBoundary.isNeumann() or rBoundary.isNeumann():
            self._boundary_helper = self._neumann_set_values_helper
        elif lBoundary.isPeriodic() or rBoundary.isPeriodic():
            assert lBoundary.isPeriodic(), 'Error: Both boundaries are required to be periodic!'
            assert rBoundary.isPeriodic(), 'Error: Both boundaries are required to be periodic!'
            self._boundary_helper = self._periodic_set_values_helper
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


    """ Boundary Helper 2D """
    def _setup_bd_2d(self):
        # TODO: currently 2D simulations always default to periodic bcs
        pass


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
        bw = self.bw

        if bw > 0:
            self.y[:, bw:-bw]   = y[:, self.ps:self.pe]
        else:
            self.y[:, :] = y[:, self.ps:self.pe]

        # make sure this runs before deform!
        self.r_dot_over_r       = 0.

        # reset ydot
        self.ydot.fill(0)

        # Deal with any boundary updates that are required by the boundary conditions
        self._boundary_helper(y)

        # do possible deformation
        if self.deformation is not None:
            self.deform(t, y)

        # Finally reset
        self._reset()


    """ Helper for dealing with periodic boundary conditions """
    def _periodic_set_values_helper(self, y):
        # Deal with left and right boundary
        self.y[:, 0:self.bw] = self.y[:, self.pCb + self.nx + self.bw]
        self.y[:, self.Cb + self.nx + self.bw] = self.y[:, self.bw + self.Cb]


    """ Combine left + right boundary helpers for Neumann and Glueing """
    def _set_values_helper(self, y):
        self._reset_boundary_approx_1d()
        self._boundary_helper_left(y)
        self._boundary_helper_right(y)


    """ Helper for glueing two patches together """
    def _glue_set_values_helper_left(self, y):
        # simply grab the two values left of the current patch
        self.y[:, 0:self.bw] = y[:, self.ps-self.bw:self.ps]

        # update boundary approximation
        # self.boundary_approx[:, 0] = np.zeros((self.n, 1))


    def _glue_set_values_helper_right(self, y):
        # simply grab the two values right of the current patch
        self.y[:, self.Cb + self.nx + self.bw] = y[:, self.pe:self.pe+self.bw]

        # update boundary approximation
        # self.boundary_approx[:, 1] = np.zeros((self.n, 1))


    """ Helper for dealing with right / left boundaries being Neumann """
    def _neumann_set_values_helper_left(self, y):
        # This setups the boundary as follows: Here || denotes the boundary
        # | y1 | y0 || y0 | y1 |
        self.y[:, 0:self.bw] = self.y[:, self.nCb + self.bw]

        # update boundary values -> don't have to change anything
        # self.boundary_approx[:, 0] = np.zeros((self.n, 1))


    def _neumann_set_values_helper_right(self, y):
        # This setups the boundary as follows: Here || denotes the boundary
        # | yN-1 | yN || yN | yN-1 |
        self.y[:, self.Cb + self.nx + self.bw] = self.y[:, self.mCb + self.nx]

        # update boundary values -> don't have to change anything
        # self.boundary_approx[:, 1] = np.zeros((self.n, 1))


    """ Implementation of no-flux boundary conditions """
    def _noflux_set_values_helper(self, y):
        # TODO ensure that we can call these separately!
        self._noflux_set_values_helper_left(y)
        self._noflux_set_values_helper_right(y)

        # update boundary values
        self._compute_boundary_approx_noflux_1d(self.bw)


    def _noflux_set_values_helper_left(self, y):
        # This setups the boundary as follows: Here || denotes the boundary
        # | y1 | y0 || y0 | y1 |
        self.y[:, 0:self.bw] = self.y[:, self.nCb + self.bw]


    def _noflux_set_values_helper_right(self, y):
        # This setups the boundary as follows: Here || denotes the boundary
        # | yN-1 | yN || yN | yN-1 |
        self.y[:, self.Cb + self.nx + self.bw] = self.y[:, self.mCb + self.nx]


    """ Implementation of dirichlet boundary conditions """
    def _dirichlet_set_values_helper(self, y):
        # use the above mentioned extrapolations
        y = self.y[:, self.bw:-self.bw]

        # Left Boundary

        # This setups the boundary as follows: Here || denotes the boundary
        # | alpha_D | alpha_D || y0 | y1 |
        mask = self.boundary.left.bc_mask()
        nask = self.boundary.left.nbc_mask()

        # in some place we need to reflect for NoBC
        self.y[:, 0:self.bw] = mask * self.boundary.left(y) + nask * self.y[:, self.nCb + self.bw]

        # Right Boundary
        # This setups the boundary as follows: Here || denotes the boundary
        # | yN-1 | yN || alpha_D | alpha_D |
        mask = self.boundary.right.bc_mask()
        nask = self.boundary.right.nbc_mask()

        self.y[:, self.Cb + self.nx + self.bw] = mask * self.boundary.right(y) + nask * self.y[:, self.mCb + self.nx]

        # setup boundary approximations
        self._compute_boundary_approx_dirichlet_1d(self.bw)


    """ Compute boundary approximations in the presence of no-flux BC in 1d.

        This function computes α_D in the case of no-flux BC:

            α_D = max[0, U(i) - 0.5[ U(i - ν ej) - U(i) ].

    """
    def _compute_boundary_approx_noflux_1d(self, bw):
        self._reset_boundary_approx_1d()

        # use the above mentioned extrapolations
        y = self.y[:, bw:-bw]

        # Here we enforce positivity!
        self.boundary_approx[:, 0] = np.maximum(0, y[:, 0]  + 0.5 * (y[:, 1]  - y[:, 0]))
        self.boundary_approx[:, 1] = np.maximum(0, y[:, -1] - 0.5 * (y[:, -2] - y[:, -1]))


    """ We just have to set the values to zero """
    def _reset_boundary_approx_1d(self):
        self.boundary_approx    = np.zeros((self.n, 2))
        self.taxis_bd_aprx      = np.zeros((self.n, 2))
        self.diffusion_bd_aprx  = np.zeros((self.n, 2))


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
        bw                  = self.bw

        if bw > 0:
            self.y[:, bw:-bw, bw:-bw]   = y[:, self.ps:self.pe]
        else:
            self.y[:, :] = y[:, self.ps:self.pe]

        # reset ydot
        self.ydot.fill(0)

        # let's define periodic boundary conditions by default for the beginning
        self.y[:, 0:bw, bw:-bw] = self.y[:, self.pCb + self.nx + bw, bw:-bw]
        self.y[:, bw:-bw, 0:bw] = self.y[:, bw:-bw, self.pCb + self.nx + bw]

        # TODO check that it's really nx here and not ny
        self.y[:, self.Cb + self.ny + bw, bw:-bw] = self.y[:, bw + self.Cb, bw:-bw]
        self.y[:, bw:-bw, self.Cb + self.ny + bw] = self.y[:, bw:-bw, bw + self.Cb]

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

