#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

# grid
from Grid import Grid

# fluxes
from FluxDiffusion import DiffusionFlux
from FluxTaxis import TaxisFlux

from utils import zeros

#
# TODO: MySQL integration for data storage
#       XML    definition of fluxes etc
#
"""
    This class implements a Taxis-Diffusion-Reaction Solver for parabolic
    partial differential equations.

    The TDR solver is introduced in A. Gerisch 2001. This solver also
    implements the required numerical methods for non-local terms, whose
    numerical analysis was introduced in A. Gerisch 2010.

    This class computes the right hand side of the following MOL-ODE.

     d y(t)[i]
    ---------  = H(t, y(t))[i].
       d t

    The right hand side is discretized in conservation form:

                        1   d
        H(U(t), i) = - ---  ∑ (F_(j)(U(t), i) - F_(j)(U(t), i - ej))
                        h  j=1

    where::
        Fj(U(t), i) is approximate average flux from Ω_(i) to Ω_(i + ej)
        through the interface of Ω_(i) and Ω_(i + ej)

    Here we distinguish between three possible right hand side terms:
        1) Diffusion

                               1   d
            H_(D)(U(t), i) =  ---  ∑ (D_(j)(U(t), i) - D_(j)(U(t), i - ej))
                               h  j=1

            where::
                D_(j)(U(t), i) is the average negative diffusion flux from
                Ω_(i) to Ω_(i + ej) through the interface Γ.

            in detail:
                                  D
                D_(j)(U(t), i) = --- (U_(i + ej) - U_(i))
                                  h

        2) Taxis

            TODO

        3) Reaction

            H_(R) = p(t, xi, Ui, C.,j)

"""
class TDR(object):
    def __init__(self, noPDEs = 1, dimensions = 1, *args, **kwargs):
        # terms
        self.version        = 'TDR-python-0.1'
        self.FNonLocal      = None # the non-local term
        self.FReac          = None # reaction terms
        self.FTrans         = None # diffusion term
        self.size           = noPDEs
        self.dimensions     = dimensions

        self.haveReactionTerms      = False
        self.haveDiffusionTerms     = False
        self.haveTaxisTerms         = True
        self.haveNonLocalTerms      = True

        # the flux term functions which are passed to the grid
        self.fluxTerms = {}

        # context access remove
        self.ctx = None

        # grid
        self.grid = None

        # values passed to solver
        self.y      = None
        self.ydot   = None

        # setup
        self._setup(*args, **kwargs)


    """ has Transport effects registered """
    def hasTransport(self):
        return self.haveTaxisTerms or self.haveNonLocalTerms


    def hasDiffusion(self):
        return self.haveDiffusionTerms


    """ compute required boundaryWidth """
    def get_bw(self):
        if self.hasTransport():
            return 2
        elif self.hasDiffusion():
            return 1
        return 0


    """ This is the entry point for the integrator """
    def __call__(self, t, y):
        # check that we dont have NaN values
        if np.isnan(np.sum(y)):
            raise ValueError("Encountered NaN in TDR update")

        # update everything
        self.update(t, y)

        # dev check!
        assert self.ydot is not None, ''

        # return the new ydot
        return self.ydot


    """ setup TDR """
    def _setup(self, *args, **kwargs):
        # Load Grid data and create Grid
        # number of patches
        nop = kwargs.pop('nop', 1)
        ngb = kwargs.pop('ngb', None)
        dX  = kwargs.pop('dX',  None)
        x0  = kwargs.pop('x0',  None)
        N   = kwargs.pop('N',   None)

        # load remaining kwargs and args

        # load transition matrices
        trans    = kwargs.pop('transitionMatrix', zeros(self.size))
        Adhtrans = kwargs.pop('AdhesionTransitionMatrix', zeros(self.size))

        # grid information
        grd = { 'nop' : nop, 'ngb' : ngb, 'dX' : dX, 'N' : N, 'x0' : x0}

        self.grid = Grid(self.size, grd, self.dimensions, bw = self.get_bw(),
                         nonLocal = np.sum(np.abs(Adhtrans))!=0, **kwargs)

        # create basic flux types
        dFlux = DiffusionFlux(self.size, self.dimensions, trans)
        tFlux = TaxisFlux(self.size,     self.dimensions, trans, Adhtrans)

        # TODO reaction

        self.fluxTerms['diffusion'] = dFlux
        self.fluxTerms['taxis'] = tFlux


    """ update between solver steps """
    def update(self, t, y):
        self.t = t
        self.y = y.reshape(self.size, self.grid.gridsize)
        self.grid.update(t, self.y)

        if self.haveReactionTerms:
            assert False, 'not implemented'
            pass

        # compute the fluxes
        for name, flux in self.fluxTerms.items():
            self.grid.apply_flux(flux)

        self.ydot = self.grid.get_ydot()


if __name__ == '__main__':
    print('Testing TDR')
    # create data object
    tdr = TDR()

    # create an initial condition for testing
    L = 10
    cellsPerUnitLength = 2**5
    n = L * cellsPerUnitLength
    h = 1. / (2**5)
    x = np.linspace(0, L, n)
    y1 = np.sin(2. * np.pi * x / L)
    e1 = (2. * np.pi / L) * np.cos(2. * np.pi * x / L)
    y2 = np.cos(2. * np.pi * x / L)
    e2 = -(2. * np.pi / L) * np.sin(2. * np.pi * x / L)

    tdr.update(0., np.vstack((y1, y2)))

    #data.set_values(np.vstack((y1, y2)))
    #data.h = h
    #data._compute_uDx()
    #data._compute_uAvx()

    #ctx = GlobalSolverContext()
    #ctx.data = data


