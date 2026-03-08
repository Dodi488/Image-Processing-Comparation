import numpy as np

def gaussian_numpy(image):
    # Define the 3x3 Gaussian kernel
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0
    
    h, w = image.shape
    # Pad the image to handle borders
    padded_img = np.pad(image, 1, mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    
    # Vectorized convolution operation
    for i in range(3):
        for j in range(3):
            output += padded_img[i:i+h, j:j+w] * kernel[i, j]
            
    return np.clip(output, 0, 255).astype(np.uint8)

def sobel_numpy(image):
    # Define Sobel X and Y kernels
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    h, w = image.shape
    padded_img = np.pad(image, 1, mode='edge')
    gx = np.zeros_like(image, dtype=np.float64)
    gy = np.zeros_like(image, dtype=np.float64)

    # Vectorized convolution for both gradients
    for i in range(3):
        for j in range(3):
            gx += padded_img[i:i+h, j:j+w] * Sx[i, j]
            gy += padded_img[i:i+h, j:j+w] * Sy[i, j]

    # Calculate magnitude: G = sqrt(Sx^2 + Sy^2)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.clip(magnitude, 0, 255).astype(np.uint8)

def median_numpy(image):
    h, w = image.shape
    padded_img = np.pad(image, 1, mode='edge')
    
    # Stack the 3x3 neighborhood into a 3D array
    neighborhood = np.zeros((h, w, 9), dtype=image.dtype)
    idx = 0
    for i in range(3):
        for j in range(3):
            neighborhood[:, :, idx] = padded_img[i:i+h, j:j+w]
            idx += 1
            
    # Calculate median along the 3rd axis (the 9 neighborhood values)
    return np.median(neighborhood, axis=2).astype(np.uint8)
