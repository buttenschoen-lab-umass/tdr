#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import unittest
import numpy as np

from tdr.Domain import Interval


class DomainTest(unittest.TestCase):

    def test_interval_construction(self):
        interval = Interval(a = 0., b = 1.)

        self.assertEqual(interval.x0, 0.)
        self.assertEqual(interval.xf, 1.)
        self.assertEqual(interval.L,  1.)
        self.assertEqual(interval.n,  6)
        self.assertEqual(interval.cellsPerUnitLength,  2**6)
        self.assertEqual(interval.M,  2**6)
        self.assertEqual(interval.h,  1./2**6)
        self.assertTrue(isinstance(interval.N, np.ndarray))
        self.assertTrue(isinstance(interval.M, np.int))
        self.assertTrue(isinstance(interval.n, np.int))


    def test_interval_xs(self):
        interval = Interval(a = 0., b = 1.)
        x  = interval.x
        self.assertEqual(np.min(x), 0.)
        self.assertEqual(np.max(x), 1.)

        xs = interval.xs()
        self.assertEqual(np.min(xs), 0.)
        self.assertEqual(np.max(xs), 1.)


    def test_interval_patches(self):
        a = 0.
        b = 2.
        nPL = 2**6
        L   = np.abs(b-a)
        Nex = int(nPL * L)

        for nop in np.arange(1, 10, 1):
            interval = Interval(a=a, b=b, nop=nop)
            self.assertEqual(interval.x0, a)
            self.assertEqual(interval.xf, b)
            self.assertLessEqual(Nex, np.sum(interval.N))
            self.assertEqual(np.sum(interval.N), int(interval.L * interval.cellsPerUnitLength))

            # now check that dX is set correctly
            Nactual = np.sum(interval.N)

            hexp = L / Nactual
            for i in range(nop):
                self.assertAlmostEqual(interval.dX[i], hexp)


if __name__ == '__main__':
    unittest.main()

