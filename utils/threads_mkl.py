import contextlib
import ctypes
import os
from ctypes.util import find_library

#os.environ['MKL_DYNAMIC']='false'
#os.environ['MKL_NUM_THREADS']='8'
os.environ['MKL_VERBOSE']='1'

# Prioritize hand-compiled OpenBLAS library over version in /usr/lib/
# from Ubuntu repos
mkl_root = os.environ['MKLROOT']

try_paths = [mkl_root + '/lib/intel64',
             find_library('mkl_rt')]

mkl_lib = None
for libpath in try_paths:
    try:
        mkl_lib = ctypes.cdll.LoadLibrary(libpath)
        break
    except OSError:
        continue
if mkl_lib is None:
    raise EnvironmentError('Could not locate an MKL shared library', 2)


def set_num_threads(n):
    """Set the current number of threads used by the MKL server."""
    # The function with capital letters accepts arguments by value!
    mkl_lib.MKL_Set_Num_Threads(int(n))


try:
    mkl_lib.MKL_Get_Max_Threads()
    def get_num_threads():
        """Get the current number of threads used by the MKL server."""
        return mkl_lib.MKL_Get_Max_Threads()
except AttributeError:
    def get_num_threads():
        """Dummy function (symbol not present in %s), returns -1."""
        return -1
    pass


@contextlib.contextmanager
def num_threads(n):
    """Temporarily changes the number of MKL threads.

    Example usage:

        print("Before: {}".format(get_num_threads()))
        with num_threads(n):
            print("In thread context: {}".format(get_num_threads()))
        print("After: {}".format(get_num_threads()))
    """
    old_n = get_num_threads()
    set_num_threads(n)
    try:
        yield
    finally:
        set_num_threads(old_n)



if __name__ == '__main__':
    import numpy as np

    print('Max threads %d.' % get_num_threads())

    x = np.random.rand(1000000)
    y = np.random.rand(1000000)
    r = np.dot(x,y)

    print('Try context manager!')
    with num_threads(4):
        np.dot(x,y)
