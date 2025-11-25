# Ciphertext Domain Super-Resolution

This directory contains the implementation of homomorphic encryption-based single image super-resolution (SISR) in the ciphertext domain.

## Overview

This implementation performs super-resolution on encrypted images using homomorphic encryption (HEAAN scheme). The code supports both single-ciphertext and multi-ciphertext approaches for processing images of different sizes.

## Project Structure

```
ciphertext_code/
├── Hear.cpp          # Main entry point - handles parameter checking, encryption initialization, and network computation
├── model.cpp         # Contains interfaces for implementing residual network operations
├── settings.cpp      # Contains network structure parameters and encryption scheme parameters
├── myUtils.cpp       # Common utility functions (memory management, image loading, mask generation, etc.)
├── testUtils.cpp     # Test functions
├── Makefile          # Build configuration
├── HEAAN/            # HEAAN homomorphic encryption library
│   ├── src/          # Source files
│   ├── lib/          # Library files
│   └── run/          # Runtime files
├── weight/           # Model weight files
└── params/           # Parameter files
```

## Key Components

### Main Files

- **Hear.cpp**: Main function entry point. It checks command-line arguments, initializes encryption parameters, loads low-resolution images, encrypts them, performs network computation, and decrypts the results.

- **model.cpp**: Implements the residual network interfaces including:
  - Convolution operations (3×3 kernels)
  - ReLU activation (polynomial approximation)
  - Sub-pixel convolution for upsampling
  - Multi-ciphertext versions of all operations
  - Fast bicubic interpolation

- **settings.cpp**: Defines network structure parameters (channel numbers, kernel sizes) and encryption scheme parameters (logq, logp, logn, slots).

- **myUtils.cpp**: Provides utility functions for:
  - Memory usage monitoring
  - Image loading and conversion
  - Weight mask generation for packing
  - Parameter initialization

## Build Instructions

### Compilation

```bash
make Hear
```

### Running

```bash
./Hear <input_image_path> <height> <width>
```

Examples:
```bash
make Hear && ./Hear lr.txt 64 64
make Hear && ./Hear my_lr.txt 32 32  # Input is upscaled size
make Hear && ./Hear my_lr.txt 64 64
make Hear && ./Hear testpit.txt 64 64
```

## Input Format

The input image should be a text file containing pixel values in row-major order. For RGB images, the format should be:
- Channel 1 (R) values
- Channel 2 (G) values  
- Channel 3 (B) values

Each pixel value should be separated by spaces, and each row should be on a new line.

## Output Format

The output is saved to `my_finaloutput.txt` containing the super-resolved image pixel values in the same format as the input.

## Network Architecture

The network follows a residual learning architecture:
1. Input convolution layer (3 channels → 64 channels)
2. 7 residual blocks (each with convolution + ReLU)
3. Output convolution layer (64 channels → 3 channels)
4. Sub-pixel convolution for upsampling (2× scale factor)

## Encryption Parameters

- **logq**: Ciphertext modulus (default: 880)
- **logp**: Message quantization parameter (default: 30)
- **logn**: Log2 of number of slots (default: 14, giving 16384 slots)
- **slots**: Number of slots = 2^logn

## Dependencies

- HEAAN homomorphic encryption library
- NTL (Number Theory Library)
- OpenMP (for parallel processing)

## Notes

- The implementation supports multi-ciphertext mode for images larger than the slot capacity
- ReLU activation is approximated using a polynomial function
- The code uses SIMD packing to process multiple pixels simultaneously
- Memory usage is monitored during execution

