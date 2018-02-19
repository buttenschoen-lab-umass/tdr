#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen


class Time(object):
    def __init__(self, tf = 1., dt = 0.1, t0 = 0.):
        self.tf = tf
        self.dt = dt
        self.t0 = t0

        self.currentTime = self.t0


    def reset(self):
        self.currentTime = self.t0


    def step(self):
        self.currentTime += self.dt


    def remaining(self):
        return self.currentTime < self.tf


    def __call__(self):
        return self.currentTime

