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
    def __init__(self, h, L, N2, R = 1., *args, **kwargs):
        self.h      = h
        self.L      = L
        self.R      = R
        self.M      = np.ceil(self.R / self.h).astype(int)
        self.N2     = int(N2)
        self.N1ext  = self.N2
        self.N1     = self.N2

        self.beta0  = kwargs.pop('beta0', 0.)
        self.betaL  = kwargs.pop('betaL', 0.)

        # check that the domain is sufficiently large!
        assert 2. * self.R < self.L, 'The domain size must be twice as large as the sensing radius!'

        self.NR     = max(1000, np.round(self.N2 * self.R).astype(int).flatten())
        self.hr     = 1. / self.NR

        self.lm     = self.M
        self.lp     = self.M + 1

        self.weights = None
        # use for non pp bc
        self.circ    = None
        # use for pp bc
        self.circFFT = None

        # the non-local operator mode
        self.mode    = kwargs.pop('mode', 'periodic')

        """ Compute integration weights """
        self._init()


    """ Functor """
    def __call__(self, u):
        return self._eval(u)


    """ Periodic evaluation """
    def _eval_pp(self, u):
        return ifft(np.tile(self.circFFT, (1, 1)) * fft(u))[0]


    # TODO: improve the functions below!
    """ No-flux evaluation """
    def _eval_noflux(self, u):
        # hmm u has shape (1, N)
        u = u[0]
        bw = 2*self.M
        #print('u:', u.shape, ' M:', self.M,' bw:',bw)
        z = ifft(np.tile(self.circFFT, (1, 1)) * fft(u))[0]
        z[:self.M]  = np.dot(self.circ,          u[:bw])[0:self.M]
        u_rhs = u[-bw:]
        rhs = -np.dot(self.circ, u_rhs[::-1])[:self.M]
        z[-self.M:] = rhs[::-1]
        return z


    """ Weakly adhesive evaluation """
    def _eval_weakly_adhesive(self, u):
        # hmm u has shape (1, N)
        u = u.flatten()
        bw = 2*self.M
        #print('u:', u.shape, ' M:', self.M,' bw:',bw)
        z = ifft(np.tile(self.circFFT, (1, 1)) * fft(u))[0]
        z[:self.M]  = np.dot(self.circ,          u[:bw])[0:self.M]
        u_rhs = u[-bw:]
        rhs = -np.dot(self.circ, u_rhs[::-1])[:self.M]
        z[-self.M:] = rhs[::-1]

        # add correction terms. At the moment we assume that we only have
        # uniform kernels
        z[:self.M]  += self.correction_lhs
        z[-self.M:] += self.correction_rhs

        return z


    """ Initialization """
    def _init(self):
        self._init_weights()
        self._init_bcs()


    """ Define the boundary interaction functions for no-flux bcs """
    def _init_no_flux(self):
        pass


    def _init_weights(self):
        if self.mode == 'periodic':
            print('Computing integration weights for periodic bcs.')
            self._init_weights_periodic()
        elif self.mode == 'no-flux':
            print('Computing integration weights for no-flux bcs.')
            self._init_weights_periodic()
            self._init_weights_noflux()
        elif self.mode == 'no-flux-reflect':
            print('Computing integration weights for no-flux reflect bcs.')
            self._init_weights_periodic()
        elif self.mode == 'weakly-adhesive' or self.mode == 'naive':
            print('Computing integration weights for weakly-adhesive bcs.')
            self._init_weights_periodic()
            self._init_weights_weakly_adhesive()
        else:
            assert False, 'Unknown non-local gradient mode %s.' % self.mode


    def _init_bcs(self):
        if self.mode == 'periodic':
            self._init_bcs_periodic()
            self._eval = self._eval_pp
        elif self.mode == 'no-flux':
            self._init_bcs_noflux()
            self._eval = self._eval_noflux
        elif self.mode == 'no-flux-reflect':
            self._eval = self._eval_pp
            self._init_bcs_periodic()
        elif self.mode == 'naive':
            self._eval = self._eval_weakly_adhesive
            self._init_bcs_weakly_adhesive()
            self.beta0 = 0.
            self.betaL = 0.
        elif self.mode == 'weakly-adhesive':
            print('Using weakly-adhesive non-local gradient mode:')
            print('\tWith β = (%.4g, %.4g).' % (self.beta0, self.betaL))
            self._eval = self._eval_weakly_adhesive
            self._init_bcs_weakly_adhesive()
        else:
            assert False, 'Unknown non-local gradient mode %s.' % self.mode

        print('NonLocalGradient ready!')


    """ Implementation details """
    def _check_guess(self, x, xi, l, lower = 0., upper = 1.):
        return ((x - xi) / self.h - l < lower) or ((x - xi) / self.h - l > upper)


    """ The uniform integration kernel """
    def _integration_kernel(self):
        OmegaM = lambda r : (-1. / (2. * self.R)) * (np.abs(r) <= self.R)
        OmegaP = lambda r : ( 1. / (2. * self.R)) * (np.abs(r) <= self.R)
        int_kernel = lambda r : np.piecewise(r, [r < 0., r > 0.], [lambda r : OmegaM(r), lambda r : OmegaP(r)])
        return int_kernel


    """ Periodic implementation """
    def _init_weights_periodic(self):
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
            # compute l1
            l1 = np.floor((x - xi) / self.h).astype(int)
            if self._check_guess(x, xi, l1):
                l1 -= 1

            assert not self._check_guess(x, xi, l1), 'NonLocalGradient weights.  Error l1!'

            # compute l2
            l2 = l1 + 1
            assert not self._check_guess(x, xi, l1, lower=-1., upper=1.), 'NonLocalGradient weights.  Error l1.2!'
            assert not self._check_guess(x, xi, l2, lower=-1., upper=1.), 'NonLocalGradient weights.  Error l2!'

            Phi1x = 1. - np.abs(x - xi - l1 * self.h) / self.h
            Phi2x = 1. - np.abs(x - xi - l2 * self.h) / self.h

            fac = 0.5 if np.abs(j) == self.NR else 1.

            self.weights[l1 + self.lm] += fac * Phi1x * Omegar
            self.weights[l2 + self.lm] += fac * Phi2x * Omegar

        self.weights *= self.hr


    def _init_bcs_periodic(self):
        # only supports pp atm
        circ = np.concatenate((self.weights[self.lm::-1], np.zeros(self.N2 - self.weights.size), self.weights[:self.lm:-1]))
        self.circFFT    = fft(circ)


    """ No-flux implementation """
    def _get_indices(self, x):
        R = self.R
        L = self.L

        # interaction with the boundary!
        f1 = lambda x : np.piecewise(x, [x < R,     x >= R],   [lambda x : R - 2. * x, -R])
        f2 = lambda x : np.piecewise(x, [x < L - R, x >= L-R], [R, lambda x : 2. * L - R - 2. * x])

        lowerIntLimit = np.floor(f1(x) / self.hr).astype(int)
        upperIntLimit = np.floor(f2(x) / self.hr).astype(int)

        idxs = None
        if x < 0.5 * R:
            idxs = np.arange(max(1, lowerIntLimit), self.NR + 1, 1)
        elif x > L - 0.5 * R:
            idxs = np.arange(lowerIntLimit, upperIntLimit + 1, 1)
        else:
            idxs = np.concatenate((np.arange(lowerIntLimit, 0, 1), np.arange(1, upperIntLimit + 1, 1)))

        return idxs.astype(int)


    def _init_weights_noflux(self):
        self.weights_bc = np.zeros((self.M, self.lm + self.lp + 1))
        xi = (self.lm + 4.5) * self.h

        # get all the x indices
        x_idxs = np.arange(0, self.M, 1)
        for i in x_idxs:
            xk = (i/self.N1) * self.L

            # integration indices
            idxs = self._get_indices(xk)

            int_kernel = self._integration_kernel()
            for j in idxs:
                r = (j/self.NR) * self.R
                assert r != 0., 'NonLocalGradient weights r cannot be zero!'
                Omegar = int_kernel(r)

                x = xi + 0.5 * self.h + r
                # compute l1
                l1 = np.floor((x - xi) / self.h).astype(int)
                if self._check_guess(x, xi, l1):
                    l1 -= 1

                assert not self._check_guess(x, xi, l1), 'NonLocalGradient weights.  Error l1!'

                # compute l2
                l2 = l1 + 1
                assert not self._check_guess(x, xi, l1, lower=-1., upper=1.), 'NonLocalGradient weights.  Error l1.2!'
                assert not self._check_guess(x, xi, l2, lower=-1., upper=1.), 'NonLocalGradient weights.  Error l2!'

                Phi1x = 1. - np.abs(x - xi - l1 * self.h) / self.h
                Phi2x = 1. - np.abs(x - xi - l2 * self.h) / self.h

                fac = 0.5 if np.abs(j) == self.NR else 1.

                #print('x:', xk,' r:', r,' Omegar:', Omegar)
                self.weights_bc[i, l1 + self.lm] += fac * Phi1x * Omegar
                self.weights_bc[i, l2 + self.lm] += fac * Phi2x * Omegar

        self.weights_bc *= self.hr


    def _init_bcs_noflux(self):
        # the middle of the domain part
        circ = np.concatenate((self.weights[self.lm::-1], np.zeros(self.N2 - self.weights.size), self.weights[:self.lm:-1]))
        self.circFFT    = fft(circ)

        # deal with the boundary weights
        NSO2    = int(self.M + 1)
        NC      = int(2 * self.M + 2 * NSO2)
        #print("NC:", NC, " NSO2:", NSO2)

        # now create a matrix we can multiply with
        circ    = np.zeros((NC, NC))

        # get all the x indices
        x_idxs = np.arange(0, self.M, 1)
        for i in x_idxs:
            circ[i + NSO2, i:i+2 * NSO2] = self.weights_bc[i, :]

        self.circ = circ[NSO2:-NSO2, NSO2:-NSO2]


    """ Weakly adhesive no-flux implementation """
    def _get_indices_naive(self, x):
        R = self.R
        L = self.L

        # interaction with the boundary!
        f1 = lambda x : np.piecewise(x, [x < R,     x >= R],   [lambda x : -x, -R])
        f2 = lambda x : np.piecewise(x, [x < L - R, x >= L-R], [R, lambda x : L - x])

        lowerIntLimit = np.floor(f1(x) / self.hr).astype(int)
        upperIntLimit = np.floor(f2(x) / self.hr).astype(int)

        # TODO test this
        idxs = np.concatenate((np.arange(lowerIntLimit, 0, 1), np.arange(1, upperIntLimit + 1, 1)))

        return idxs.astype(int)


    def _init_weights_weakly_adhesive(self):
        self.weights_bc = np.zeros((self.M, self.lm + self.lp + 1))
        xi = (self.lm + 4.5) * self.h

        # get all the x indices
        x_idxs = np.arange(0, self.M, 1)
        for i in x_idxs:
            xk = (i/self.N1) * self.L

            # integration indices
            idxs = self._get_indices_naive(xk)

            int_kernel = self._integration_kernel()
            for j in idxs:
                r = (j/self.NR) * self.R
                assert r != 0., 'NonLocalGradient weights r cannot be zero!'
                Omegar = int_kernel(r)

                x = xi + 0.5 * self.h + r
                # compute l1
                l1 = np.floor((x - xi) / self.h).astype(int)
                if self._check_guess(x, xi, l1):
                    l1 -= 1

                assert not self._check_guess(x, xi, l1), 'NonLocalGradient weights.  Error l1!'

                # compute l2
                l2 = l1 + 1
                assert not self._check_guess(x, xi, l1, lower=-1., upper=1.), 'NonLocalGradient weights.  Error l1.2!'
                assert not self._check_guess(x, xi, l2, lower=-1., upper=1.), 'NonLocalGradient weights.  Error l2!'

                Phi1x = 1. - np.abs(x - xi - l1 * self.h) / self.h
                Phi2x = 1. - np.abs(x - xi - l2 * self.h) / self.h

                fac = 0.5 if np.abs(j) == self.NR else 1.

                #print('x:', xk,' r:', r,' Omegar:', Omegar)
                self.weights_bc[i, l1 + self.lm] += fac * Phi1x * Omegar
                self.weights_bc[i, l2 + self.lm] += fac * Phi2x * Omegar

        self.weights_bc *= self.hr


    def _init_bcs_weakly_adhesive(self):
        # the middle of the domain part
        circ = np.concatenate((self.weights[self.lm::-1], np.zeros(self.N2 - self.weights.size), self.weights[:self.lm:-1]))
        self.circFFT    = fft(circ)

        # deal with the boundary weights
        NSO2    = int(self.M + 1)
        NC      = int(2 * self.M + 2 * NSO2)
        #print("NC:", NC, " NSO2:", NSO2)

        # now create a matrix we can multiply with
        circ    = np.zeros((NC, NC))

        # get all the x indices
        x_idxs = np.arange(0, self.M, 1)
        for i in x_idxs:
            circ[i + NSO2, i:i+2 * NSO2] = self.weights_bc[i, :]

        self.circ = circ[NSO2:-NSO2, NSO2:-NSO2]

        # compute correction terms
        xs = np.linspace(0, self.L - self.h, self.N2)

        self.correction_lhs = self.beta0 * (xs[:self.M] - self.R) / (2. * self.R)
        self.correction_rhs = self.betaL * (self.R - (self.L - xs[-self.M:])) / (2. * self.R)

        assert self.correction_lhs.size == self.M, 'Size is correction is incorrect!'
        assert self.correction_rhs.size == self.M, 'Size is correction is incorrect!'


