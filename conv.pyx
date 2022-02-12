

# tag: numpy
# You can ignore the previous line.
# It's for internal testing of the cython documentation.

import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np

# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
np.import_array()

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.int

# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
#ctypedef np.int_t DTYPE_t


# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
#

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def conv_maxon(
    np.ndarray[np.uint8_t, ndim=2] values_array, 
    np.ndarray[np.uint8_t, ndim=1] kernel_x, 
    np.ndarray[np.uint8_t, ndim=1] kernel_y, 
    np.ndarray[np.int8_t, ndim=1] kernel_val, 
    int maxon, 
    int xs, 
    int ys
):

    assert values_array.dtype == np.uint8 and kernel_x.dtype == np.uint8 and kernel_y.dtype == np.uint8 and kernel_val.dtype == np.int8
 
    cdef np.ndarray[np.uint8_t, ndim=2] target_array = np.zeros([xs, ys], dtype=np.uint8)
    cdef int ker_len = kernel_x.size
    cdef int counter
    cdef Py_ssize_t i, j, row

    for i in range(xs):
        for j in range(ys):

            counter = 0
            for row in range(ker_len):
                if values_array[i + kernel_x[row], j + kernel_y[row]] != 0:
                    counter += kernel_val[row]
                    if counter >= maxon:
                        target_array[i, j] = 1
                        break

    return target_array







@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def conv_maxon2(
    np.ndarray[np.uint8_t, ndim=2] values_array, 
    np.ndarray[np.uint8_t, ndim=1] kernel_x, 
    np.ndarray[np.uint8_t, ndim=1] kernel_y, 
    np.ndarray[np.int8_t, ndim=1] kernel_val, 
    int maxon, 
    int xs, 
    int ys
):

    assert values_array.dtype == np.uint8 and kernel_x.dtype == np.uint8 and kernel_y.dtype == np.uint8 and kernel_val.dtype == np.int8
 
    cdef np.ndarray[np.uint8_t, ndim=2] target_array = np.zeros([xs, ys], dtype=np.uint8)
    cdef int ker_len = kernel_x.size
    cdef int counter, value
    cdef Py_ssize_t i, j, row, kx, ky


    for row in range(ker_len):
        kx = kernel_x[row]
        ky = kernel_y[row]
        value = kernel_val[row]

        for i in range(xs):
            for j in range(ys):
                if values_array[i + kx, j + ky] != 0:
                    target_array[i, j] += value
    

    #cdef int tmp
    #for i in range(xs):
    #    for j in range(ys):
    #        tmp = target_array[i, j]
    #        target_array[i, j] = 1 if tmp >= maxon else 0


    return target_array
























