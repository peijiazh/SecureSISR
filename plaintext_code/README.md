# Plaintext Domain Super-Resolution

This directory contains the implementation of single image super-resolution (SISR) in the plaintext domain using PyTorch.

## Overview

This implementation trains and evaluates a residual network for image super-resolution. The model uses polynomial ReLU activation functions that are compatible with homomorphic encryption operations.

## Project Structure

```
plaintext_code/
├── model.py          # Network architecture definition
├── train.py          # Training script
├── eval.py           # Evaluation script for testing datasets
├── Dataset.py        # Dataset loading utilities
├── Relu_version.py   # Polynomial ReLU activation functions
├── testprocess.py    # Test processing utilities
├── test_pit.py       # Pixel shuffle testing
├── readmodel.py      # Model weight extraction
├── compute_once.py    # Single computation utilities
├── drawpit.py        # Visualization utilities
└── weight/           # Trained model weights
```

## Key Components

### Main Files

- **model.py**: Defines the network architecture:
  - `Network`: Main super-resolution network with residual blocks
  - `Conv_ReLU_Block`: Residual block with convolution and polynomial ReLU
  - Input layer: 3 channels → 64 channels
  - 7 residual layers
  - Output layer: 64 channels → 3 channels

- **train.py**: Training script with:
  - SGD optimizer with momentum
  - Learning rate scheduling
  - Gradient clipping
  - Checkpoint saving
  - Multi-GPU support

- **eval.py**: Evaluation script that:
  - Tests on standard datasets (Set5, Set14, etc.)
  - Computes PSNR and SSIM metrics
  - Supports multiple scale factors (2×, 3×, 4×)
  - Compares with bicubic interpolation baseline

- **Relu_version.py**: Contains polynomial approximations of ReLU:
  - Various polynomial ReLU functions (tri_fit, four_fit)
  - Compatible with homomorphic encryption operations
  - Different approximation levels for different ranges

## Installation

### Requirements

```bash
pip install torch torchvision
pip install numpy scipy
pip install h5py
pip install pytorch-ssim
pip install matlab.engine  # Optional, for YCbCr conversion
```

## Usage

### Training

```bash
python train.py --batch_size 256 --epochs 50 --lr 0.1 --cuda --gpus "0,1"
```

Key arguments:
- `--batch_size`: Training batch size (default: 256)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.1)
- `--cuda`: Use GPU acceleration
- `--gpus`: GPU IDs to use (default: "2,3")
- `--resume`: Path to checkpoint to resume from
- `--pretrained`: Path to pretrained model

### Evaluation

```bash
python eval.py --model model_testrgb/model_epoch_50.pth --dataset data/test_data/Set5_rgb --gpus "0"
```

Key arguments:
- `--model`: Path to trained model checkpoint
- `--dataset`: Test dataset path
- `--gpus`: GPU ID to use

### Extract Model Weights

```bash
python readmodel.py --model model_testrgb/model_epoch_50.pth --cuda
```

This extracts the trained weights and saves them in the `weight/` directory for use in the ciphertext implementation.

## Dataset Format

The training script expects HDF5 files with the following structure:
- `data`: Low-resolution images (shape: [N, H, W, C])
- `label`: High-resolution images (shape: [N, H, W, C])

Example dataset preparation:
```python
import h5py
import numpy as np

with h5py.File('train.h5', 'w') as f:
    f.create_dataset('data', data=lr_images)
    f.create_dataset('label', data=hr_images)
```

## Network Architecture

The network uses:
- **Input layer**: Conv2d(3, 64, kernel_size=3, padding=1)
- **Residual blocks**: 7 blocks, each with Conv2d(64, 64, kernel_size=3, padding=1) + Polynomial ReLU
- **Output layer**: Conv2d(64, 3, kernel_size=3, padding=1)
- **Residual connection**: Adds input to output

## Polynomial ReLU

The activation function uses a polynomial approximation:
```
relu(x) ≈ a₃x³ + a₂x² + a₁x + a₀
```

This allows the same activation to be used in both plaintext training and ciphertext inference.

## Training Tips

1. **Learning Rate**: Start with 0.1 and reduce by 10× every 10 epochs
2. **Gradient Clipping**: Use 0.4 to stabilize training
3. **Batch Size**: Larger batch sizes (256+) work well
4. **Data Augmentation**: Consider random flips and rotations
5. **Weight Initialization**: Uses He initialization by default

## Evaluation Metrics

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): Range [0,1], higher is better

Both metrics are computed on the Y channel of YCbCr color space.

## Output

- Trained models are saved in `model_testrgb/` directory
- Model weights for ciphertext inference are saved in `weight/` directory
- Evaluation results show PSNR and SSIM for each scale factor

## Notes

- The polynomial ReLU is designed to match the ciphertext implementation
- Training uses MSE loss (sum reduction)
- The model supports arbitrary scale factors through sub-pixel convolution
- RGB images are processed directly (no YCbCr conversion during training)

## License

See the main README for license information.

