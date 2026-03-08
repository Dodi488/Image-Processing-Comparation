import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def gaussian_cython(double[:, :] image):
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] output = np.zeros((h, w), dtype=np.float64)
    cdef double[:, :] output_view = output
    
    # Gaussian kernel
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
            output_view[i, j] = val
            
    return np.asarray(output)
