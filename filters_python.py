def apply_kernel_python(image, kernel):
    h, w = len(image), len(image[0])
    k_size = len(kernel)
    pad = k_size // 2
    
    # Create empty output image
    output = [[0.0 for _ in range(w)] for _ in range(h)]
    
    for i in range(pad, h - pad):
        for j in range(pad, w - pad):
            val = 0.0
            for ki in range(k_size):
                for kj in range(k_size):
                    val += image[i - pad + ki][j - pad + kj] * kernel[ki][kj]
            output[i][j] = val
    return output

def gaussian_python(image):
    # 3x3 Gaussian Kernel
    kernel = [
        [1/16, 2/16, 1/16],
        [2/16, 4/16, 2/16],
        [1/16, 2/16, 1/16]
    ]
    return apply_kernel_python(image, kernel)

# Note: For Sobel, you would apply an X kernel and a Y kernel, then calculate the gradient magnitude using math.sqrt().
