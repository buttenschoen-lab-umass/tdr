#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division

import numpy as np
from scipy.fftpack import fft, ifft

"""
    This class implements the computation of the so called non-local gradient.

    The non-local gradient is given by

                           R
        K[u(x, t)](x, t) = ∫ g(u(x + r, t)) Ω(r) dr
                          -R

    where::
        R    is the cell's sensing radius
        g(∙) describes the nature of the force
        Ω(∙) describes the force direction


    Implementation Details:
        At the moment this class only implements the non-local gradient for
        periodic boundary conditions.

        The integral above is solved using the FFT. For the mathematical
        details of the implementation see A. Gerisch 2010.

    Usage: The class implements a functor.

        h  = 0.1
        L  = 10.
        N2 = L * (1. / h)
        R  = 1.
        G  = NonLocalGradient(h, L, N2, R)
        u  = np.sin(...)
        grad = G(u)

    Todo: 1) Make weights settable again
          2) Implement other boundary conditions

"""
class NonLocalGradient:
    def __init__(self, h, L, N2, R = 1.):
        self.h = h
        self.L = L
        self.N1 = 0
        self.N1ext = 0
        self.N2 = np.asscalar(N2)

        self.R = R
        self.NR = max(1000, np.round(self.N2 * self.R).flatten())

        self.M = np.ceil(self.R / self.h).astype(int)
        self.lm = self.M
        self.lp = self.M + 1

        # second dimension
        self.km = self.M
        self.kp = self.M

        self.weights = None
        self.circ    = None
        self.circFFT = None

        print('Init non-local gradient')
        self._init_weights()
        self._init_bcs()
        print('Non-local gradient ready!')


    """ Functor """
    def __call__(self, u):
        return self._eval(u)[0]


    """ Implementation details """
    def _check_guess(self, x, xi, l, lower = 0., upper = 1.):
        return ((x - xi) / self.h - l < lower) or ((x - xi) / self.h - l > upper)


    def _get_bilinear_interp_params(self, evalAt, h, ci, cj, lp, kp, eps = 1e-8):
        l   = np.floor(evalAt[0]  / h)
        d1  = (evalAt[0] - h * l) / h
        k   = np.floor(evalAt[1]  / h)
        d2  = (evalAt[1] - h * k) / h

        # check that d1 and d2 are between (0, 1)
        thresfac = 100
        if d1 < 0:
            if d1 > -thresfac*eps:
                d1 = 0
            else:
                assert False, 'Bad value of d1!'

        if d1 > 1:
            if d1 - 1 < thresfac*eps:
                d1 = 1
            else:
                assert False, 'Bad value of d1!'

        if d2 < 0:
            if d2 > -thresfac*eps:
                d2 = 0
            else:
                assert False, 'Bad value of d2!'

        if d2 > 1:
            if d2 - 1 < thresfac*eps:
                d2 = 1
            else:
                assert False, 'Bad value of d2!'

        # ensure limits on (k, l)
        if ci + l <= 0:
            assert False, 'l too small'

        if cj + k <= 0:
            assert False, 'k too small'

        if l > lp:
            assert False, 'l too large'

        if l == lp:
            if d1 < thresfac * eps:
                l = l - 1
                d1 = 1
            else:
                assert False, 'l out of bounds'

        if k > kp:
            assert False, 'k too large'

        if k == kp:
            if d2 < thresfac * eps:
                k = k - 1
                d2 = 1
            else:
                assert False, 'k out of bounds'

        # test computed values
        if np.abs(evalAt[0] - h * (l + d1)) > 1e-15:
            assert False, 'x error'

        if np.abs(evalAt[1] - h * (k + d2)) > 1e-15:
            assert False, 'y error'

        return l, k, d1, d2


    def _init_weights(self):
        """
            This function computes the integration weights using a midpoint
            composite using the formula presented in section 3.1.3 of Gerisch
            2010
        """
        OmegaH = lambda r : (1. / (np.pi * self.R * self.R)) * (np.abs(r) <= self.R)

        # set weighttol to the lattice spacing
        self.weighttol = self.h

        cj = self.km + 1
        ci = self.lm + 1

        converged = False
        iters     = 0
        maxiter   = 10

        while not converged:
            iters += 1

            # initialize
            Wx1 = np.zeros(self.km + self.kp + 1, self.lm + self.lp + 1)
            Wx2 = Wx1

            DR = self.R / self.NR
            mIdxs = np.arange(1, self.NR)
            for m in mIdxs:
                rm      = (m * self.R) / self.NR
                facRm   = rm * OmegaH(rm) / self.NR

                curNTheta = np.max(10, np.round(2. * np.pi * rm / DR))
                curDTheta = 2. * np.pi / curNTheta

                nIdxs = np.arange(1, curNTheta)
                for n in nIdxs:
                    thetan = curDTheta * n
                    etan   = np.asarray([np.cos(thetan), np.sin(thetan)])
                    evalAt = rm * etan + np.asarray([self.h / 2, 0])

                    l, k, d1, d2 = self._get_bilinear_interp_params(evalAt, self.h, ci, cj, self.lp, self.kp)
                    gamma  = facRm * etan * curDTheta

                    # update the matrix!
                    Wx1[cj+k, ci+l]     += gamma[0] * (1. - d1) * (1. - d2)
                    Wx2[cj+k, ci+l]     += gamma[1] * (1. - d1) * (1. - d2)

                    Wx1[cj+k, ci+l+1]   += gamma[0] * d1 * (1. - d2)
                    Wx2[cj+k, ci+l+1]   += gamma[1] * d1 * (1. - d2)

                    Wx1[cj+k+1, ci+l]   += gamma[0] * (1. - d1) * d2
                    Wx2[cj+k+1, ci+l]   += gamma[1] * (1. - d1) * d2

                    Wx1[cj+k+1, ci+l+1] += gamma[0] * d1 * d2
                    Wx2[cj+k+1, ci+l+1] += gamma[1] * d1 * d2

            if iters == 1:
                Wx1old = Wx1
                Wx2old = Wx2

                self.NR = 2 * self.NR
            else:
                wdiff1 = Wx1old - Wx1
                wdiff2 = Wx2old - Wx2
                wdiff = np.max(np.concatenate(wdiff1, wdiff2))
                if wdiff < self.weighttol:
                    converged = True
                elif iters > maxiter:
                    assert False, 'Maximum number of iterations!'
                else:
                    Wx1old = Wx1
                    Wx2old = Wx2
                    self.NR = 2 * self.NR


    def _init_bcs(self):
        # only supports pp atm
        self.circ = np.concatenate((self.weights[self.lm::-1],
                               np.zeros(self.N2 - self.weights.size),
                               self.weights[:self.lm:-1]))

        self.N1ext = self.N2
        self.N1 = self.N2
        self.circFFT = fft(self.circ)


    def _eval(self, u):
        return ifft(np.tile(self.circFFT, (1, 1)) * fft(u))





