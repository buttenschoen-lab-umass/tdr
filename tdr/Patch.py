#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np

from tdr.helpers import cartesian
from tdr.Data import Data
from tdr.NonLocalGradient import NonLocalGradient
from tdr.Boundary import DomainBoundary


class Patch(object):
    """
        A patch of a domain.

        Parameters
        ----------
        n               : int
                          Number of PDEs
        patchId         : int
                          Id of the corresponding patch
        ngb             : dict
                          Identifying the boundary patches
        x0              : ndarray
                          The lower corner of the patch
        xf              : ndarray
                          The upper corner of the patch
        dX              : ndarray
                          Array of step lengths in each dimension
        N               : ndarray
                          Array giving size of partition set
        boundaryWidth   : int
                          Width of points to add for boundary conditions
        nonLocal        : bool
                          Whether a non-local gradient should be constructed
        **kwargs        : dict
                        These are forwarded to Data


        Data Layout
        -----------
        1) First the row of grid cells with the smallest x-values, then the row
           just above.

        2) Within each row of grid values start with the ones with the smallest
           y-values.


        Attributes
        ----------

        TODO

    """
    def __init__(self, *args, **kwargs):
        self.x0     = kwargs.pop('x0')
        self.xf     = kwargs.pop('xf', None)
        self.N      = np.asarray(kwargs.get('N'), np.int)
        self.dim    = self.N.size
        self.ngb    = kwargs.pop('ngb', DomainBoundary(self.dim))

        # reference to the non-local gradient
        self.nonLocalGradient   = None

        # data for the patch
        self.data = None

        # array holding the cell centers
        self.xc = None

        # setup
        self._setup(*args, **kwargs)

        # uses information only saved in data
        self.shape  = self.N * self.dX

        # if xf is not set guess
        if self.xf is None:
            self.xf     = self.x0 + self.shape

        # Print status
        print('Creating patch(%d) with lower corner %s and upper corner %s.' %
              (self.patchId, self.x0, self.xf))


    def __str__(self):
        return 'Patch(Id = %d, d = %d; x0 = %s; xf = %s; dX = %s; n = %d; N = %s; se=[%d, %d]).' % \
                (self.patchId, self.dim, self.x0, self.xf, self.dX, self.n, self.N, self.ps, self.pe)


    def __repr__(self):
        return self.__str__()


    """ Redirect attribute gets to the data sub-object """
    def __getattr__(self, name):
        return getattr(self.data, name)


    """ Public methods """
    def update(self, t, y):
        # must run these before
        self.data.set_values(t, y)
        self.data.compute_face_data()


    def length(self):
        return self.N * self.dX


    def endPoints(self):
        return self.x0 + self.length()


    def apply_flux(self, flux, t):
        flux(self, t)


    def step_size(self):
        return self.dX


    """ Computes the centers of the small volumes in the domain """
    def cellCenters(self):
        return self.x0 + cartesian(self.xc)


    """ The integer shape of the patch """
    def get_shape(self):
        return self.N


    """ Resizing of the domain """
    # only for 1D so far
    # TODO: optimize the recomp of cell centers!
    def growPatchRight(self):
        # easy simply add N
        self.N += 1

        # redo cell centers
        self._setup_cell_centers()


    def growPatchLeft(self):
        # grow to the right but first move left most point
        self.x0 -= self.dX
        self.growPatchRight()


    def shrinkPatchRight(self):
        # reduce N by one
        self.N -= 1
        self._setup_cell_centers()


    def shrinkPatchLeft(self):
        # shift x0 to the right
        self.x0 += self.dX
        self.shrinkPatchRight()


    """ Setup internal data structures! """
    def _setup(self, *args, **kwargs):
        nonLocal = kwargs.pop('nonLocal', False)
        if nonLocal:
            self._setup_nonlocal(*args, **kwargs)

        self.data = Data(ngb=self.ngb, *args, **kwargs)

        # now we can set the end
        self.data.pe = self.ps + self.memory_size()

        # build cell centers
        self._setup_cell_centers()


    """ Creates patch center coordinates - centered around the origin """
    def _setup_cell_centers(self):
        xcs = []
        for i in range(self.N.size):
            xcs.append((np.arange(1, self.N[i] + 1, 1) - 0.5) * self.dX[i])

        self.xc = np.array(xcs)


    """ Non-local term special setup """
    def _nonlocal_mode(self, *args, **kwargs):
        assert self.ngb.left.name == self.ngb.right.name, \
                'Different boundary conditions are not supported for the non-local term!'

        if self.ngb.left.name == 'Periodic':
            return 'periodic'
        # TODO differentiate between the different types of no-flux bc
        elif self.ngb.left.name == 'Neumann' or self.ngb.left.name == 'NoFlux':
            return kwargs.pop('nonLocalMode', 'no-flux')
        else:
            assert False, 'Unknown boundary type %s!' % self.ngb.left.name


    def _setup_nonlocal(self, *args, **kwargs):
        assert self.dX.size == 1, 'Non-local operator support only available for a single patch!'
        mode     = self._nonlocal_mode(*args, **kwargs)
        int_mode = kwargs.pop('int_mode', 'uniform')
        beta0 = kwargs.pop('beta0', 0.)
        betaL = kwargs.pop('betaL', 0.)

        self.nonLocalGradient = NonLocalGradient(self.dX[0], self.shape[0],
                                                 self.N[0], mode=mode,
                                                 kernel = int_mode,
                                                 beta0=beta0, betaL=betaL,
                                                 *args, **kwargs)


