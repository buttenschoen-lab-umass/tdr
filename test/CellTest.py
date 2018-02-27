#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen

from Cell.Simulator import Simulator


if __name__ == '__main__':
    sim = Simulator()
    sim.init()
    sim.simulate()


