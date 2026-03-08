import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

# Import all filter functions from their respective modules
from filters_python import gaussian_python, sobel_python, median_python
from filters_numpy import gaussian_numpy, sobel_numpy, median_numpy
from filters_cython import gaussian_cython, sobel_cython, median_cython

def measure_time(func, image, name):
    """Measures and prints the execution time of a given function."""
    start = time.time()
    result = func(image)
    end = time.time()
    exec_time = end - start
    print(f"{name:<25}: {exec_time:.4f} seconds")
    return result, exec_time

def main():
    # 1. Load Image and convert to grayscale
    img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load image. Please ensure 'test_image.jpg' is in the directory.")
        return

    # Convert image to standard Python lists for the Pure Python implementation
    img_list = img.tolist()
    # Cast image to float64 for Cython compatibility
    img_float = img.astype(np.float64)

    # Dictionary to store the results for visualization
    results = {}
    times = [[], [], []]

    print("\n--- Gaussian Filter Performance ---")
    results['Gauss_Py'], _ = measure_time(gaussian_python, img_list, "Pure Python")
    times[0].append(_)
    results['Gauss_Np'], _ = measure_time(gaussian_numpy, img, "NumPy")
    times[1].append(_)
    results['Gauss_Cy'], _ = measure_time(gaussian_cython, img_float, "NumPy + Cython")
    times[2].append(_)

    print("\n--- Sobel Filter Performance ---")
    results['Sobel_Py'], _ = measure_time(sobel_python, img_list, "Pure Python")
    times[0].append(_)
    results['Sobel_Np'], _ = measure_time(sobel_numpy, img, "NumPy")
    times[1].append(_)
    results['Sobel_Cy'], _ = measure_time(sobel_cython, img_float, "NumPy + Cython")
    times[2].append(_)

    print("\n--- Median Filter Performance ---")
    results['Median_Py'], _ = measure_time(median_python, img_list, "Pure Python")
    times[0].append(_)
    results['Median_Np'], _ = measure_time(median_numpy, img, "NumPy")
    times[1].append(_)
    results['Median_Cy'], _ = measure_time(median_cython, img_float, "NumPy + Cython")
    times[2].append(_)

    # 4. Visualize Results
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    fig.suptitle('Image Processing Filters: Performance Comparison', fontsize=16)

    # Define layout mapping for the plot
    plot_layout = [
        ('Gaussian', results['Gauss_Py'], results['Gauss_Np'], results['Gauss_Cy']),
        ('Sobel', results['Sobel_Py'], results['Sobel_Np'], results['Sobel_Cy']),
        ('Median', results['Median_Py'], results['Median_Np'], results['Median_Cy'])
    ]

    for row, (filter_name, res_py, res_np, res_cy) in enumerate(plot_layout):
        axes[row, 0].imshow(img, cmap='gray')
        axes[row, 0].set_title('Original')
        axes[row, 0].axis('off')

        axes[row, 1].imshow(res_py, cmap='gray')
        axes[row, 1].set_title(f'{filter_name} (Pure Python)\n{times[0][row]:.4f}')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(res_np, cmap='gray')
        axes[row, 2].set_title(f'{filter_name} (NumPy)\n{times[1][row]:.4f}')
        axes[row, 2].axis('off')

        axes[row, 3].imshow(res_cy, cmap='gray')
        axes[row, 3].set_title(f'{filter_name} (Cython)\n{times[2][row]:.4f}')
        axes[row, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the output image as required by visual results
    plt.savefig('comparative_visualization.png')
    plt.show()

if __name__ == "__main__":
    main()

