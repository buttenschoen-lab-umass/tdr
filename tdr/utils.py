#!/usr/bin/python
# -*- coding: utf-8 -*-
# Author: Andreas Buttenschoen
import numpy as np

# move to utils
def PeriodicGradient(u):
    dx = np.zeros_like(u)

    dx[1:-1]    = (u[2:] - u[:-2])
    dx[0]       = (u[1]  - u[-1])
    dx[-1]      = (u[0]  - u[-2])

    return dx


# the vanLeer limiter
VanLeer = lambda r : (r + np.abs(r)) / (1. + np.abs(r))


def asarray(obj):
    arr = np.asarray(obj)
    if arr.size == 1:
        arr = arr.reshape(1, 1)
    return arr


def apply_along_column(functions, arr):
    """
    Apply a 1D-array to functions column wise to data in arr

    Parameters
    ----------
    functions: 1-D array of lambdas that take n arguments.

    arr : ndarray
        This is the data to which function is applied to column-wise. The
        number of rows must match the length of functions.

    Returns
    -------
    out : ndarray
        2-D array of shape of arr. With the result of function is returned in
        row i.
    """
    outarray = np.empty_like(arr, arr.dtype)

    # create rows
    rows = []
    for row in range(arr.shape[0]):
        rows.append(arr[row, :])

    # apply the function row wise
    for row in range(arr.shape[0]):
        outarray[row, :] = functions[row](*rows)

    return outarray


# https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def zeros(size):
    return np.zeros(size*size).reshape(size, size)


