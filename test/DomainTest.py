#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import unittest
import numpy as np

from tdr.Domain import Interval
from tdr.Domain import Square


class IntervalTest(unittest.TestCase):

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
        self.assertEqual(len(interval.N.shape), 1)


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
            self.assertEqual(len(interval.N.shape), 1)
            self.assertEqual(interval.N.size, nop)

            # now check that dX is set correctly
            Nactual = np.sum(interval.N)

            hexp = L / Nactual
            for i in range(nop):
                self.assertAlmostEqual(interval.dX[i], hexp)


class SquareTest(unittest.TestCase):

    def test_square_construction(self):
        square = Square(a = 0., b = 1., c = 0., d = 1.)

        self.assertEqual(square.x0, 0.)
        self.assertEqual(square.xf, 1.)
        self.assertTrue(np.all(square.L == np.array([1.,1.])))
        self.assertEqual(square.n,  8)
        self.assertEqual(square.cellsPerUnitLength,  2**8)
        self.assertEqual(square.M,  2**8)
        self.assertEqual(square.h,  1./2**8)
        self.assertTrue(isinstance(square.N, np.ndarray))
        self.assertTrue(isinstance(square.M, np.int))
        self.assertTrue(isinstance(square.n, np.int))
        self.assertEqual(len(square.N.shape), 2)


    def test_square_xs(self):
        square = Square(a = 0., b = 1., c = 1., d = 2.)
        x  = square.x
        self.assertEqual(np.min(x), 0.)
        self.assertEqual(np.max(x), 1.)

        y = square.y
        self.assertEqual(np.min(y), 1.)
        self.assertEqual(np.max(y), 2.)

        xs = square.xs()
        self.assertEqual(np.min(xs), 0.)
        self.assertEqual(np.max(xs), 1.)

        ys = square.ys()
        self.assertEqual(np.min(ys), 1.)
        self.assertEqual(np.max(ys), 2.)


if __name__ == '__main__':
    unittest.main()

