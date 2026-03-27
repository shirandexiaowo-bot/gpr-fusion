# Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network

Official PyTorch implementation for the paper: "Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network".

This repository provides a two-stage sequential deep learning framework for multi-scene Ground Penetrating Radar (GPR) data fusion. The framework integrates Cycle-GAN for unpaired data translation and a Transformer-based network for high-quality data fusion.

## Framework Overview

The framework consists of two sequential stages designed to bridge the gap between different GPR sensing scenarios:

### Stage 1: Cycle-GAN (Domain Alignment)

Task: Unpaired GPR data translation and time-frequency domain alignment.

Key Features:

- Training on unpaired datasets (no direct pixel-to-pixel labels required).
- Alignment of time-domain wave characteristics and frequency-domain distributions.
- Generates pseudo high-frequency GPR images from low-frequency inputs.

### Stage 2: Transformer Fusion Network (Adaptive Synthesis)

Task: Adaptive weighted GPR data fusion.

Key Features:

- Fuses Stage 1 pseudo-results with real high-frequency GPR data.
- Captures long-range dependencies via Transformer blocks.
- Outputs high-resolution, high-fidelity fused GPR profiles.

## Project Structure

- `cycle_gan/`: Stage 1: Domain alignment module
- `datasets/gpr/`: Training/Testing GPR data
- `options/`: Hyperparameter configurations
- `requirements.txt`: Stage 1 dependencies
- `README.md`: Detailed Stage 1 guide
- `transformer/`: Stage 2: Fusion module
- `data/`: Demo and fusion input data
- `requirements.txt`: Stage 2 dependencies
- `README.md`: Detailed Stage 2 guide
- `LICENSE`: MIT License

## Quick Start

1. Requirements
	* Python >= 3.7
	* PyTorch >= 1.8
	* CUDA-enabled GPU (Recommended)
2. Stage 1: Cycle-GAN Training & Inference
	* Bash: cd cycle_gan && pip install -r requirements.txt
3. Stage 2: Transformer Fusion
	* Bash: cd ../transformer && pip install -r requirements.txt
	* Place Stage 1 outputs into ./data/
	* Run the fusion inference (demo data included)
		+ Example command: python test_fusion.py

## Data Preparation

Module | Data Path | Description
---------|-------------|-------------
Cycle-GAN | ./cycle_gan/datasets/gpr/ | Unpaired L-freq and H-freq GPR B-scans
Transformer | ./transformer/data/ | Stage 1 results + Real H-freq data

Note: We provide demo data in the transformer/data/ directory. You can run the Stage 2 inference directly to verify the installation and view the fusion performance.

## Citation

If you find this code or research helpful, please cite our paper:

@article{YourName2026GPR,
	title={Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network},
	author={Huaxiang Yin, Xihong Cui, et al.},
	journal={Computers & Geosciences},
	year={2026}
}

## License

This project is licensed under the MIT License - see the LICENSE file for details.
