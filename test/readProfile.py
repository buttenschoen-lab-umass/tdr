#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division, absolute_import
import pstats

#
# TODO load file from command line argument
#
# To run a profile execute: python3 -m cProfile -s cumtime -o tdr2d_vanLeer.out test/tdrProfile2D.py
#

p = pstats.Stats('tdr2d.out')
p.sort_stats('cumulative').print_stats(50)


