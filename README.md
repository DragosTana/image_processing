# Image Processing

This repository contains an implementation of a 2D convolution algorithm for image processing. The algorithm has been parallelized using both OpenMP and CUDA to achieve enhanced performance and exploit multi-core CPUs as well as GPU acceleration.

# Getting Started

## Prerequisites

Before you can use and compile the code in this repository, you'll need to have the following installed on your system:

* C++ compiler with OpenMP support (e.g., GCC)
* NVIDIA GPU with CUDA support (for GPU acceleration)

## Compilation

1. Clone this repository to your local machine.
2. Navigate to the repository's directory.
3. Run `make` to use the provided Makefile
## Usage

### Input
Before running the program, ensure you have the input image you want to process. Supported image formats include JPEG, PNG, and BMP. Some test images are available in the images directory

### Running the Program
To run use `./convolution <image_path> <kernel_type> <algorithm>`

- `<image_path>` path of the image
- `<kernel_typ>` can be: _blur_
- `<algorithm>`  can be: _cuda_, _seq_, _omp_

Replace input_image.jpg with the path to your input image. For example:

`./convolution images/lenna_gray_512.jpg blur cuda`

## Performance

The performance of the convolution algorithm can vary based on factors such as the size of the input image, size of the kernel,  the number of CPU cores available, and the GPU specifications. As a general guideline, larger images and powerful GPUs tend to yield better performance gains from parallelization.

# TODO

Add different kernels

