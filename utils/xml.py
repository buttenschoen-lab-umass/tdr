#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen


def hasName(xmlNode, name):
    return xmlNode.tag == name


def isParameter(xmlNode):
    return hasName(xmlNode, 'Parameter')


def isFields(xmlNode):
    return hasName(xmlNode, 'Fields')


def isField(xmlNode):
    return hasName(xmlNode, 'Field')


def isFieldView(xmlNode):
    return hasName(xmlNode, 'FieldView')


def isReaction(xmlNode):
    return hasName(xmlNode, 'Reaction')


def isNoise(xmlNode):
    return hasName(xmlNode, 'Noise')


def isInitialCondition(xmlNode):
    return hasName(xmlNode, 'InitialCondition')


def isFieldGroup(xmlNode):
    return hasName(xmlNode, 'FieldGroup')


def isTime(xmlNode):
    return hasName(xmlNode, 'Time')


def isDomain(xmlNode):
    return hasName(xmlNode, 'Interval')


def isCells(xmlNode):
    return hasName(xmlNode, 'Cells')


def isCell(xmlNode):
    return hasName(xmlNode, 'Cell')


def isSphericalCell(xmlNode):
    return hasName(xmlNode, 'SphericalCell')


def isCellSpherical(xmlNode):
    return hasName(xmlNode, 'SphericalCell')


def isBoundingBox(xmlNode):
    return hasName(xmlNode, 'BoundingBox')


def isBoundary(xmlNode):
    return hasName(xmlNode, 'Boundary')


def isDomainBoundary(xmlNode):
    return hasName(xmlNode, 'DomainBoundary')


def isNoiseGenerator(xmlNode):
    return hasName(xmlNode, 'NoiseGenerator')


def isAdhesionBond(xmlNode):
    return hasName(xmlNode, 'AdhesionBond')
