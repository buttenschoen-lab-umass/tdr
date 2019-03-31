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
        self.assertTrue(isinstance(interval.N, np.int))
        self.assertTrue(isinstance(interval.M, np.int))
        self.assertTrue(isinstance(interval.n, np.int))
        self.assertTrue(isinstance(interval.cellsPerUnitLength, np.int))


    def test_interval_xs(self):
        interval = Interval(a = 0., b = 1.)
        x  = interval.x
        self.assertEqual(np.min(x), 0.)
        self.assertEqual(np.max(x), 1.)

        xs = interval.xs()
        self.assertEqual(np.min(xs), 0.)
        self.assertEqual(np.max(xs), 1.)


if __name__ == '__main__':
    unittest.main()



