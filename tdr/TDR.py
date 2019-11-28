#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function

import numpy as np

# domain support
from tdr.Domain import Interval

# grid
from tdr.Grid import Grid

# fluxes
from tdr.FluxDiffusion import DiffusionFlux
from tdr.FluxTaxis import TaxisFlux
from tdr.FluxAdvection import AdvectionFlux
from tdr.FluxReaction import ReactionFlux
from tdr.FluxDilution import DilutionFlux

from tdr.helpers import zeros, asarray, offdiagonal


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


        4) Advection

                       1
            H_(A) = - --- Finish me
                       h

"""
class TDR(object):

    # keywords used for transition matrices
    # Transition is used for both taxis and diffusion
    __transition__ = 'transitionMatrix'

    # used for advection terms -> mainly hyperbolic systems
    __advection__  = 'transport'

    # Used for coefficients for non-local taxis terms
    __adhesion__   = 'AdhesionTransitionMatrix'

    # used for reaction terms
    __reaction__   = 'reactionTerms'

    # TODO get rid of this!
    __flux_names__ = {__transition__ : ['diffusion', 'taxis'],
                      __advection__  : ['advection'],
                      __reaction__   : ['reaction'],
                      __adhesion__   : ['taxis']}


    def __init__(self, *args, **kwargs):
        # terms
        self.version        = 'TDR-python-0.2'
        self.size           = kwargs.pop('noPDEs', 1)
        self.dimensions     = None
        self.bw             = kwargs.pop('bw', 0)

        # for easy checking of requirements
        self.haveReactionTerms      = False
        self.haveAdvectionTerms     = False
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


    def hasAdvection(self):
        return self.haveAdvectionTerms


    """ The next two are equivalent at the moment for easy checking for any
        deformations that may be occurring.
    """
    def hasDilution(self):
        return self.haveDilutionTerms


    def hasDeformation(self):
        return self.haveDilutionTerms


    """ compute required boundaryWidth """
    def get_bw(self):
        # TEMP FIGURE THIS OUT!!
        if self.hasTransport() or self.hasDiffusion() or self.hasAdvection():
            return np.max(2, self.bw)
        #elif self.hasDiffusion():
        #    return np.max(1, self.bw)

        # TEMP improve handling in tdr.Data!
        return np.max(2, self.bw)


    """ TODO These functions have to be implemented via the solout callback!  """
    def _check_solution(self, t, y):
        # check that we dont have NaN values
        if np.any(np.isnan(y)):
            raise ValueError("Encountered NaN in TDR update at time: %.4g" % t)


    def _check_ydot(self, t, y):
        # check that we dont have NaN values
        if np.any(np.isnan(y)):
            raise ValueError("Encountered NaN in TDR ydot computation at time: %.4g" % t)


    """ This is the entry point for the integrator """
    def __call__(self, t, y, rpar, ipar):
        if self.debug: self._check_solution(t, y)

        # update everything
        self.update(t, y)

        # dev check!
        assert self.ydot is not None, 'In TDR update ydot is None!'

        # also debug check ydot
        if self.debug: self._check_ydot(t, self.ydot)

        # return the new ydot
        return self.ydot


    """ This is the entry point for numerical bifurcation analysis """
    def compute(self, y, *args, **kwargs):
        """ compute
            - y: ndarray
            - *args: could contain time information -> ignored currently.
            - **kwargs expects dictionary with transition matrices
        """
        #print('p:',kwargs)
        #print('fluxTerms:', self.fluxTerms)
        # update flux parameters
        for tname, tvalue in kwargs.items():
            # lookup corresponding flux(s)
            flux_names = self.__flux_names__[tname]

            # have to deal with adhesion differently
            if tname == self.__adhesion__:
                flux = self.fluxTerms[flux_names[0]]
                flux.update_adhtrans(tvalue)
            else:
                for flux_name in flux_names:
                    flux = self.fluxTerms[flux_name]
                    flux.update_trans(tvalue)

        # update everything
        self.update(0, y)

        # dev check!
        assert self.ydot is not None, 'In TDR update ydot is None!'

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
        dom = kwargs.pop('domain', Interval(0, 1))

        # set time to initial time
        self.t = kwargs.pop('t0', 0.)

        # set dimension
        self.dimensions = dom.dimensions

        # save domain
        self.dom = dom
        print('TDR: registered domain %s.' % self.dom)

        # load transition matrices
        trans      = asarray(kwargs.pop(self.__transition__, zeros(self.size)))
        advection  = asarray(kwargs.pop(self.__advection__,  zeros(self.size)))
        Adhtrans   = asarray(kwargs.pop(self.__adhesion__,   zeros(self.size)))
        reaction   = asarray(kwargs.pop(self.__reaction__,   np.full(self.size, None)), matrix=False)

        # These should be shared among all players on the domain!
        deformation = asarray(kwargs.pop('r',                       None))
        dr          = asarray(kwargs.pop('dr',                      None))

        self.haveDiffusionTerms = np.any(np.diagonal(trans)  != 0)
        self.haveReactionTerms  = np.any(reaction            != None)
        self.haveNonLocalTerms  = np.any(Adhtrans            != 0)
        self.haveTaxisTerms     = np.any(offdiagonal(trans)  != 0)
        self.haveDilutionTerms  = np.any(deformation         != None)
        self.haveAdvectionTerms = np.any(advection           != 0)

        if self.hasDeformation():
            # make sure we both have r and dr
            assert np.any(dr != None), 'The time derivative of the domain deformation must be provided!'

            # TODO: do this differently
            # this is only for house-keeping
            # self.dom.setup_deformation(deformation)

        # create a Grid
        self.grid = Grid(self.size, dom, bw=self.get_bw(),
                         nonLocal = self.hasNonLocal(), r=deformation, dr=dr,
                         *args, **kwargs)

        # create basic flux types
        if self.hasDiffusion():
            # if we have deformation support need to update diffusion coefficient
            dFlux_kwargs = {'boundary' : dom.bd}

            if self.hasDeformation():
                # setup cFunctional
                L0 = self.dom.L
                cFunctional = lambda t, y : trans / (L0 * deformation[0,0](t, y))**2
                dFlux_kwargs['cFunctional'] = cFunctional

            dFlux = DiffusionFlux(self.size, self.dimensions, trans, **dFlux_kwargs)
            self.fluxTerms['diffusion'] = dFlux


        if self.hasTransport():
            tFlux = TaxisFlux(self.size, self.dimensions, trans, Adhtrans,
                              nonLocal=self.hasNonLocal(), boundary=dom.bd)
            self.fluxTerms['taxis'] = tFlux


        if self.hasAdvection():
            aFlux = AdvectionFlux(self.size, self.dimensions, advection,
                                  boundary=dom.bd)
            self.fluxTerms['advection'] = aFlux


        if self.hasReaction():
            rFlux = ReactionFlux(self.size, self.dimensions, reaction)
            self.fluxTerms['reaction'] = rFlux


        if self.hasDilution():
            dFlux = DilutionFlux(self.size, self.dimensions, deformation)
            self.fluxTerms['dilution'] = dFlux


        # sort the flux terms
        self.fluxes = sorted(self.fluxTerms.values(),
                             key=lambda flux : flux.priority(), reverse=True)


        # print fluxes
        print('Registered fluxes:', end=' ')
        for flux in self.fluxes:
            print(flux, end='; ')
        print('', end='\n')


    """ update between solver steps """
    def update(self, t, y):
        self.t = t

        # TODO Can i get rid of this potential copy here?
        # Is this even really a copy? Most of the time its not!
        self.y = self.reshape(y)

        # update the grid!
        self.grid.update(t, self.y)

        # For the moment only update fluxes if deformation is set
        if self.hasDeformation():
            for flux in self.fluxes:
                # update potential time-depending coefficients in the fluxes
                flux.update(t)

        # compute the fluxes
        for flux in self.fluxes:
            self.grid.apply_flux(flux, t)

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


    """ output """
    def __str__(self):
        rstr = 'TDR(PDEs = %d; dim = %d; fluxes = %d; bw = %d) has:\n' % \
                (self.size, self.dimensions, len(self.fluxes), self.bw)

        if self.haveDiffusionTerms: rstr += '\tdiffusion term.'
        if self.haveReactionTerms: rstr += '\treaction term.'
        if self.haveNonLocalTerms: rstr += '\tnon-local term.'
        if self.haveDilutionTerms: rstr += '\tdilution term.'
        if self.haveAdvectionTerms: rstr += '\tadvection term.'
        if self.haveTaxisTerms: rstr += '\ttaxis term.'

        return rstr


    def __repr__(self):
        return 'TDR(PDEs = %d; dim = %d; fluxes = %d; bw = %d.' % \
                (self.size, self.dimensions, len(self.fluxes), self.bw)

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


