#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

# domain support
from tdr.Domain import Interval

# grid
from tdr.Grid import Grid

# fluxes
from tdr.FluxDiffusion import DiffusionFlux
from tdr.FluxTaxis import TaxisFlux
from tdr.FluxReaction import ReactionFlux
from tdr.FluxDilution import DilutionFlux

from tdr.utils import zeros, asarray, offdiagonal

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

                                1   d
            H_(T)(U(t), i) = - ---  ∑ (T_(j)(U(t), i) - T_(j)(U(t), i - ej))
                                h  j=1

            where::
                T_(j)(U(t), i) is the average of the tactic flux through the
                interface of Ω_(i) and Ω_(i + ej)

                T_(j)(U(t), i) = max{0, v_(ij)} S_(j)^(+)(U, i) + min{0, v_(ij)} S_(j)^(-)(U, i)

                where::
                    v_(ij) is the approximate average of the tactic velocity
                    through the interface of Ω_(i) and Ω_(i + ej)

                    S_(j)^(+/-)(U, i) are the state interpolants

                                      / U_i + 0.5 * Phi(r_(ij))(U_i - U_(i - ej))
                    S_(j)^(+)(U, i) = |
                                      \ U_i, if U_i = U_(i - ej)


                                      / U_(i + ej) + 0.5 * Phi(r^(-1)_(ij))(U_(i + ej) - U_(i + 2ej))
                    S_(j)^(-)(U, i) = |
                                      \ U_(i + ej), if U_(i + ej) = U_(i + 2ej)

                    where::
                        Phi() is the flux limiter. Below we use the so called vanLeer flux limiter.

                        r_(ij) is the smoothness monitor defined by:

                                   U_(i + ej) - U_i
                        r_(ij) =  -----------------
                                   U_i - U_(i - ej)

        3) Reaction

            H_(R) = p(t, xi, Ui, C.,j)

"""
class TDR(object):
    def __init__(self, *args, **kwargs):
        # terms
        self.version        = 'TDR-python-0.1'
        self.size           = kwargs.pop('noPDEs', 1)
        self.dimensions     = None
        self.bw             = kwargs.pop('bw', 0)

        # for easy checking of requirements
        self.haveReactionTerms      = False
        self.haveDiffusionTerms     = False
        self.haveTaxisTerms         = False
        self.haveNonLocalTerms      = False
        self.haveDilutionTerms      = False

        # the flux term functions which are passed to the grid
        self.fluxTerms  = {}

        # fluxes sorted by priority
        self.fluxes     = None

        # context access remove
        self.ctx        = None

        # store domain
        self.dom        = None

        # grid
        self.grid       = None

        # values passed to solver
        self.y          = None
        self.ydot       = None

        # whether or not to do step wise error checking
        self.debug      = kwargs.pop('debug', False)

        # setup
        self._setup(*args, **kwargs)


    """ check whether the TDR is autonomous """
    def isAutonomous(self):
        return not self.hasDeformation()


    """ has Transport effects registered """
    def hasTransport(self):
        return self.haveTaxisTerms or self.haveNonLocalTerms


    def hasNonLocal(self):
        return self.haveNonLocalTerms


    def hasDiffusion(self):
        return self.haveDiffusionTerms


    def hasReaction(self):
        return self.haveReactionTerms


    """ The next two are equivalent at the moment for easy checking for any
        deformations that may be occurings.
    """
    def hasDilution(self):
        return self.haveDilutionTerms


    def hasDeformation(self):
        return self.haveDilutionTerms


    """ compute required boundaryWidth """
    def get_bw(self):
        # TEMP FIGURE THIS OUT!!
        if self.hasTransport() or self.hasDiffusion():
            return np.max(2, self.bw)
        #elif self.hasDiffusion():
        #    return np.max(1, self.bw)

        return self.bw


    def _check_solution(self, t, y):
        # check that we dont have NaN values
        if np.any(np.isnan(y)):
            raise ValueError("Encountered NaN in TDR update at time: %.4g" % t)

        if np.any(y < 0.):
            raise ValueError("Encountered non-positive values in TDR update at time: %.4g" % t)

        if np.any(np.abs(y) > 1.e5):
            raise ValueError("Encountered too large values in TDR update at time: %.4g" % t)


    """ This is the entry point for the integrator """
    def __call__(self, t, y):
        if self.debug:
            self._check_solution(t, y)

        # update everything
        self.update(t, y)

        # dev check!
        assert self.ydot is not None, ''

        # return the new ydot
        return self.ydot


    """ get number of patches """
    def _get_nops(self, requested_nop):
        if self.dimensions == 1:
            return 1

        return requested_nop


    """ setup TDR """
    def _setup(self, *args, **kwargs):
        # Load Grid data and create Grid
        # number of patches
        nop = self._get_nops(kwargs.pop('nop', 1))
        dom = kwargs.pop('domain', Interval(0, 1))
        ngb = np.asarray(dom.bd).reshape(1)
        dX  = asarray(dom.dX)
        x0  = asarray(dom.origin())
        N   = asarray(dom.N, np.int)

        # set time to initial time
        self.t = kwargs.pop('t0', 0.)

        # set dimension
        self.dimensions = dom.dimensions()

        # save domain
        self.dom = dom
        print('TDR: registered domain %s.' % self.dom)

        # load transition matrices
        trans    = asarray(kwargs.pop('transitionMatrix',           zeros(self.size)))
        Adhtrans = asarray(kwargs.pop('AdhesionTransitionMatrix',   zeros(self.size)))
        reaction = asarray(kwargs.pop('reactionTerms',              np.full(self.size, None)))

        # These should be shared among all players on the domain!
        deformation = asarray(kwargs.pop('r',                       None))
        dr          = asarray(kwargs.pop('dr',                      None))

        self.haveDiffusionTerms = np.any(np.diagonal(trans) != 0)
        self.haveReactionTerms  = np.any(reaction != None)
        self.haveNonLocalTerms  = np.any(Adhtrans != 0)
        self.haveTaxisTerms     = np.any(offdiagonal(trans) != 0)
        self.haveDilutionTerms  = np.any(deformation != None)

        if self.hasDeformation():
            # make sure we both have r and dr
            assert np.any(dr != None), 'The time derivative of the domain deformation must be provided!'

            # TODO: do this differently
            # this is only for house-keeping
            # self.dom.setup_deformation(deformation)

        # grid information
        grd = { 'nop' : nop, 'ngb' : ngb, 'dX' : dX, 'N' : N, 'x0' : x0}

        self.grid = Grid(self.size, grd, self.dimensions, bw=self.get_bw(),
                         nonLocal = self.hasNonLocal(), r=deformation, dr=dr,
                         *args, **kwargs)

        # create basic flux types
        if self.hasDiffusion():
            # if we have deformation support need to update diffusion coefficient
            dFlux_kwargs = {'boundary' : dom.bd}

            if self.hasDeformation():
                # setup cFunctional
                L0 = self.dom.L
                cFunctional = lambda t : trans / (L0 * deformation[0,0](t))**2
                dFlux_kwargs['cFunctional'] = cFunctional

            dFlux = DiffusionFlux(self.size, self.dimensions, trans, **dFlux_kwargs)
            self.fluxTerms['diffusion'] = dFlux


        if self.hasTransport():
            tFlux = TaxisFlux(self.size, self.dimensions, trans, Adhtrans,
                              nonLocal=self.hasNonLocal(), boundary=dom.bd)
            self.fluxTerms['taxis'] = tFlux


        if self.hasReaction():
            rFlux = ReactionFlux(self.size, self.dimensions, reaction)
            self.fluxTerms['reaction'] = rFlux


        if self.hasDilution():
            dFlux = DilutionFlux(self.size, self.dimensions, deformation)
            self.fluxTerms['dilution'] = dFlux


        # sort the flux terms
        self.fluxes = sorted(self.fluxTerms.values(),
                             key=lambda flux : flux.priority(), reverse=True)


    """ update between solver steps """
    def update(self, t, y):
        self.t = t
        # TODO Can i get rid of this potential copy here?
        self.y = self.reshape(y)

        self.grid.update(t, self.y)

        # TODO: FIXME
        # just to keep information up to date
        # self.dom.deform(t)

        # For the moment only update fluxes if deformation is set
        if self.hasDeformation():
            for flux in self.fluxes:
                # update potential time-depending coefficients in the fluxes
                flux.update(t)

        # compute the fluxes
        for flux in self.fluxes:
            self.grid.apply_flux(flux)

        self.ydot = self.grid.get_ydot()


    """ reshape the data """
    def reshape(self, y):
        new_shape = np.insert(self.grid.shape(), 0, self.size)
        return y.reshape(new_shape)


    """ Resize the domain """
    def resizeDomain(self):
        # change domain object
        # need to update grid
        pass


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


