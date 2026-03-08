import numpy as np

def gaussian_numpy(image):
    # 3x3 Gaussian Kernel
    kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0
    
    h, w = image.shape
    pad = 1
    padded_img = np.pad(image, pad, mode='edge')
    output = np.zeros_like(image, dtype=np.float64)
    
    # Vectorized convolution
    for i in range(3):
        for j in range(3):
            output += padded_img[i:i+h, j:j+w] * kernel[i, j]
            
    return np.clip(output, 0, 255).astype(np.uint8)

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
            
    # Calculate median along the 3rd axis
    return np.median(neighborhood, axis=2).astype(np.uint8)
