# Image-Processing-Comparation

## Objective
This project implements three common image processing filters (Gaussian, Sobel, and Median) to analyze computational performance across three different approaches: Pure Python, NumPy, and NumPy + Cython.

## Setup Instructions
1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Compile the CPython extension:**
To run the CPython optimized filters, you must compile the C-extension firsts:

  ```
  python setup.py build_ext --inplace
  ```

## Running the Project
Place an image in the root directory and run the main script:

  ```
  python main.py
  ```

This will load the image, convert it to grayscale, apply the filters using all three methods, display the execution times and visualize the filtered images.
