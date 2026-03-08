import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

from filters_python import gaussian_python
from filters_numpy import gaussian_numpy
from filters_cython import gaussian_cython

def measure_time(func, image, name):
    start = time.time()
    result = func(image)
    end = time.time()
    exec_time = end - start
    print(f"{name} Execution Time: {exec_time:.4f} seconds")
    return result, exec_time

def main():
    # 1. Load Image and convert to grayscale
    # Replace 'test_image.jpg' with your actual image file
    img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image.")
        return

    print("--- Gaussian Filter Performance ---")
    # Convert to standard Python lists for the Pure Python implementation
    img_list = img.tolist()
    
    _, time_py = measure_time(gaussian_python, img_list, "Pure Python")
    res_np, time_np = measure_time(gaussian_numpy, img, "NumPy")
    res_cy, time_cy = measure_time(gaussian_cython, img.astype(np.float64), "NumPy + Cython")

    # 4. Visualize Results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Original\n{time_py:.4f}s')
    
    axes[1].imshow(res_np, cmap='gray')
    axes[1].set_title(f'NumPy Filtered\n{time_np:.4f}s')
    
    axes[2].imshow(res_cy, cmap='gray')
    axes[2].set_title(f'Cython Filtered\n{time_cy:.4f}s')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
