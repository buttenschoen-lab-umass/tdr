#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
from __future__ import print_function, division, absolute_import
import os, argparse
import pstats

#
# TODO load file from command line argument
#
# To run a profile execute: python3 -m cProfile -s cumtime -o tdr2d_vanLeer.out test/tdrProfile2D.py
#

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Display profile created with cProfile')
    parser.add_argument('-f', '--file')
    parser.add_argument('-n', '--lines', dest='lines', default=50)
    args = parser.parse_args()

    assert os.path.isfile(args.file), '%s can not be found!' % args.file

    p = pstats.Stats(args.file)
    p.sort_stats('tottime').print_stats(args.lines)


