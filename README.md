# ICL-Noise-UNet: In-Context Learning with Noise Modulation for Ultrasound Segmentation

This project implements ICL-Noise-UNet, a U-Net-based architecture enhanced with in-context learning and noise modulation for ultrasound image segmentation. The model leverages contextual information from similar images and incorporates noise maps (residual and local variance) to improve segmentation performance on noisy ultrasound data.

## Features

- **ICL-Noise-UNet Architecture**: A custom U-Net with context integration and noise modulation blocks.
- **In-Context Learning**: Uses context images and masks to guide segmentation of target images.
- **Noise Modulation**: Applies residual noise and local variance maps to enhance feature representations.
- **Model Comparison**: Scripts to evaluate and compare multiple segmentation models (e.g., nn-UNet, UNETR, Swin-UNet, MultiverSegNet).
- **Datasets**: Supports ultrasound datasets like BUSI, BUSBRA, CAMUS, and JNU.
- **Training and Testing**: End-to-end training with PyTorch Lightning, including validation and testing phases.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ICL-Noise_UNet.git
   cd ICL-Noise_UNet