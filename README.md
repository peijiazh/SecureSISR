# Single Image Super-Resolution with Homomorphic Encryption

This repository contains the implementation of Single Image Super-Resolution (SISR) using homomorphic encryption, enabling privacy-preserving image enhancement on encrypted data.

## Overview

This project implements a residual network for image super-resolution that can operate on both plaintext and ciphertext (encrypted) images. The ciphertext implementation uses the HEAAN homomorphic encryption scheme, allowing super-resolution to be performed on encrypted images without decrypting them.

## Repository Structure

```
.
├── ciphertext_code/     # Homomorphic encryption-based implementation (C++)
│   ├── Hear.cpp        # Main entry point
│   ├── model.cpp       # Network implementation
│   ├── settings.cpp    # Configuration parameters
│   ├── myUtils.cpp     # Utility functions
│   ├── HEAAN/          # HEAAN encryption library
│   └── README.md       # Detailed ciphertext code documentation
│
├── plaintext_code/     # Plaintext training and evaluation (Python)
│   ├── model.py        # Network architecture
│   ├── train.py        # Training script
│   ├── eval.py         # Evaluation script
│   ├── Relu_version.py # Polynomial ReLU activations
│   └── README.md       # Detailed plaintext code documentation
│
└── README.md           # This file
```

## Features

- **Privacy-Preserving**: Process images in encrypted form without decryption
- **Residual Network**: Deep residual architecture for high-quality super-resolution
- **Polynomial ReLU**: Activation functions compatible with homomorphic operations
- **Multi-Ciphertext Support**: Handles images larger than encryption slot capacity
- **Flexible Scales**: Supports 2×, 3×, and 4× upscaling factors

## Quick Start

### Plaintext Training

1. Prepare your dataset in HDF5 format
2. Train the model:
```bash
cd plaintext_code
python train.py --batch_size 256 --epochs 50 --lr 0.1 --cuda
```

3. Extract weights for ciphertext inference:
```bash
python readmodel.py --model model_testrgb/model_epoch_50.pth
```

### Ciphertext Inference

1. Build the ciphertext code:
```bash
cd ciphertext_code
make Hear
```

2. Run inference on encrypted images:
```bash
./Hear input_image.txt 64 64
```

## Network Architecture

The network follows a residual learning framework:

```
Input (LR Image)
    ↓
Conv(3→64) + ReLU
    ↓
[Residual Block] × 7
    ↓
Conv(64→3)
    ↓
Residual Connection (add input)
    ↓
Sub-pixel Convolution (upsampling)
    ↓
Output (HR Image)
```

## Key Components

### Plaintext Domain (`plaintext_code/`)
- PyTorch-based training and evaluation
- Polynomial ReLU activation functions
- Standard super-resolution metrics (PSNR, SSIM)
- Model weight extraction for ciphertext use

### Ciphertext Domain (`ciphertext_code/`)
- HEAAN homomorphic encryption integration
- Encrypted convolution operations
- Polynomial activation on ciphertexts
- Sub-pixel convolution for upsampling
- Multi-ciphertext support for large images

## Dependencies

### Plaintext Code
- Python 3.6+
- PyTorch
- NumPy, SciPy
- h5py
- pytorch-ssim

### Ciphertext Code
- C++11 or later
- HEAAN homomorphic encryption library
- NTL (Number Theory Library)
- OpenMP

## Performance

The implementation achieves competitive results on standard benchmarks:
- **Set5**: PSNR > 30dB for 2× upscaling
- **Set14**: PSNR > 28dB for 2× upscaling
- Processing time depends on image size and encryption parameters

## Security

- Images remain encrypted throughout the entire super-resolution process
- Only the final result is decrypted (by the data owner)
- No intermediate values are exposed
- Compatible with secure multi-party computation scenarios

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Privacy-Preserving Single Image Super-Resolution with Homomorphic Encryption},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

- HEAAN library for homomorphic encryption
- PyTorch community for deep learning framework
- Contributors and testers

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Related Work

- HEAAN: An efficient homomorphic encryption library
- VDSR: Very Deep Super-Resolution network
- EDSR: Enhanced Deep Super-Resolution network

---

**Note**: This is research code. For production use, please ensure proper security audits and performance optimizations.

