import math

def apply_kernel_python(image, kernel):
    # Get image dimensions
    h, w = len(image), len(image[0])
    k_size = len(kernel)
    pad = k_size // 2
    
    # Create an empty output image initialized with zeros
    output = [[0.0 for _ in range(w)] for _ in range(h)]
    
    # Iterate through every pixel, ignoring the borders based on padding
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            val = 0.0
            # Apply the kernel to the current pixel's neighborhood
            for ki in range(k_size):
                for kj in range(k_size):
                    val += image[i - pad + ki][j - pad + kj] * kernel[ki][kj]
            output[i][j] = val
    return output

def gaussian_python(image):
    # Standard 3x3 Gaussian kernel for noise reduction
    kernel = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    # Apply the kernel and return the result
    return apply_kernel_python(image, kernel)

def sobel_python(image):
    # Sobel kernels for X and Y directions
    Sx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Sy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    
    # Compute gradients in both directions
    gx = apply_kernel_python(image, Sx)
    gy = apply_kernel_python(image, Sy)
    
    h, w = len(image), len(image[0])
    output = [[0 for _ in range(w)] for _ in range(h)]
    
    # Calculate the gradient magnitude for edge detection
    for i in range(h):
        for j in range(w):
            val = math.sqrt(gx[i][j]**2 + gy[i][j]**2)
            # Clip the value to a maximum of 255 for standard 8-bit images
            output[i][j] = min(255, int(val))
    return output

def median_python(image):
    h, w = len(image), len(image[0])
    output = [[0 for _ in range(w)] for _ in range(h)]
    
    # Iterate through the image, ignoring a 1-pixel border
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Extract the 3x3 neighborhood into a 1D list
            neighborhood = [
                image[i-1][j-1], image[i-1][j], image[i-1][j+1],
                image[i][j-1],   image[i][j],   image[i][j+1],
                image[i+1][j-1], image[i+1][j], image[i+1][j+1]
            ]
            # Sort the values to find the median
            neighborhood.sort()
            # Replace the center pixel with the median (the 5th item, index 4)
            output[i][j] = neighborhood[4]
    return output
