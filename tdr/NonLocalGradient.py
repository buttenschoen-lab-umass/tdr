#!/usr/bin/python
# -*- coding: utf-8 -*-
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
        self.N2 = N2

        self.R = R
        self.NR = max(1000, np.round(self.N2 * self.R))

        self.M = np.ceil(self.R / self.h).astype(int)
        self.lm = self.M
        self.lp = self.M + 1

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


    def _init_weights(self):
        """
            This function computes the integration weights using a midpoint
            composite using the formula presented in section 3.1.3 of Gerisch
            2010
        """
        OmegaM = lambda r : (-1. / (2. * self.R)) * (np.abs(r) <= self.R)
        OmegaP = lambda r : ( 1. / (2. * self.R)) * (np.abs(r) <= self.R)

        self.weights = np.zeros(self.lm+self.lp+1)
        xi = (self.lm + 4.5) * self.h
        idxs = np.concatenate((np.arange(-self.NR, 0, 1),
                               np.arange(1, self.NR + 1, 1)))

        for j in idxs:
            r = (j/self.NR)*self.R
            if r < 0:
                Omegar = OmegaM(r)
            elif r > 0:
                Omegar = OmegaP(r)
            else:
                assert False, 'r is %.2g!. This shouldnt happen' % r

            x = xi + 0.5 * self.h + r
            l1 = np.floor((x - xi) / self.h).astype(int)
            if self._check_guess(x, xi, l1):
                l1 -= 1

            if self._check_guess(x, xi, l1):
                assert False, 'Error l1'

            l2 = l1 + 1
            if self._check_guess(x, xi, l1, lower = -1., upper = 1.):
                assert False, 'Error l1.2'
            if self._check_guess(x, xi, l2, lower = -1., upper = 1.):
                assert False, 'Error l2'

            Phi1x = 1. - np.abs(x - xi - l1 * self.h) / self.h
            Phi2x = 1. - np.abs(x - xi - l2 * self.h) / self.h

            if np.abs(j) == self.NR:
                fac = 0.5
            else:
                fac = 1.

            self.weights[l1 + self.lm] += fac * Phi1x * Omegar
            self.weights[l2 + self.lm] += fac * Phi2x * Omegar

        self.weights /= self.NR


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





