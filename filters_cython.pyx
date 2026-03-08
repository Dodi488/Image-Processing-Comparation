import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Disable bounds checking and wraparound for maximum C-level speed
@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_cython(double[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] output = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] output_view = output
    
    # Initialize the 3x3 Gaussian kernel statically
    cdef double kernel[3][3]
    kernel[0][0] = 1.0/16; kernel[0][1] = 2.0/16; kernel[0][2] = 1.0/16
    kernel[1][0] = 2.0/16; kernel[1][1] = 4.0/16; kernel[1][2] = 2.0/16
    kernel[2][0] = 1.0/16; kernel[2][1] = 2.0/16; kernel[2][2] = 1.0/16

    cdef int i, j, ki, kj
    cdef double val
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            val = 0.0
            for ki in range(3):
                for kj in range(3):
                    val += image[i - 1 + ki][j - 1 + kj] * kernel[ki][kj]
            output_view[i, j] = <unsigned char>val
            
    return np.asarray(output)

@cython.boundscheck(False)
@cython.wraparound(False)
def sobel_cython(double[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] output = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] output_view = output

    # Initialize Sobel X kernel
    cdef double Sx[3][3]
    Sx[0][0] = -1; Sx[0][1] = 0; Sx[0][2] = 1
    Sx[1][0] = -2; Sx[1][1] = 0; Sx[1][2] = 2
    Sx[2][0] = -1; Sx[2][1] = 0; Sx[2][2] = 1

    # Initialize Sobel Y kernel
    cdef double Sy[3][3]
    Sy[0][0] = -1; Sy[0][1] = -2; Sy[0][2] = -1
    Sy[1][0] =  0; Sy[1][1] =  0; Sy[1][2] =  0
    Sy[2][0] =  1; Sy[2][1] =  2; Sy[2][2] =  1

    cdef int i, j, ki, kj
    cdef double gx, gy, val

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            gx = 0.0
            gy = 0.0
            for ki in range(3):
                for kj in range(3):
                    gx += image[i - 1 + ki][j - 1 + kj] * Sx[ki][kj]
                    gy += image[i - 1 + ki][j - 1 + kj] * Sy[ki][kj]
            
            # Compute gradient magnitude
            val = sqrt(gx*gx + gy*gy)
            if val > 255: val = 255
            output_view[i, j] = <unsigned char>val
            
    return np.asarray(output)

@cython.boundscheck(False)
@cython.wraparound(False)
def median_cython(double[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef np.ndarray[np.uint8_t, ndim=2] output = np.zeros((h, w), dtype=np.uint8)
    cdef unsigned char[:, :] output_view = output

    cdef int i, j, k, l
    cdef double window[9]
    cdef double temp

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Extract the 3x3 neighborhood into a static C array
            window[0] = image[i-1][j-1]; window[1] = image[i-1][j]; window[2] = image[i-1][j+1]
            window[3] = image[i][j-1];   window[4] = image[i][j];   window[5] = image[i][j+1]
            window[6] = image[i+1][j-1]; window[7] = image[i+1][j]; window[8] = image[i+1][j+1]

            # Perform a fast inline bubble sort for the 9 elements
            for k in range(8):
                for l in range(8 - k):
                    if window[l] > window[l+1]:
                        temp = window[l]
                        window[l] = window[l+1]
                        window[l+1] = temp

            # Set the center pixel to the median value (index 4)
            output_view[i, j] = <unsigned char>window[4]
            
    return np.asarray(output)
