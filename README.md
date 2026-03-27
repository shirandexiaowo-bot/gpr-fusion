Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer NetworkOfficial PyTorch implementation for the paper: "Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network".This repository provides a two-stage sequential deep learning framework for multi-scene Ground Penetrating Radar (GPR) data fusion. The framework integrates Cycle-GAN for unpaired data translation and a Transformer-based network for high-quality data fusion.🌟 Framework OverviewThe framework consists of two sequential stages designed to bridge the gap between different GPR sensing scenarios:Stage 1: Cycle-GAN (Domain Alignment)Task: Unpaired GPR data translation and time-frequency domain alignment.Key Features:Training on unpaired datasets (no direct pixel-to-pixel labels required).Alignment of time-domain wave characteristics and frequency-domain distributions.Generates pseudo high-frequency GPR images from low-frequency inputs.Stage 2: Transformer Fusion Network (Adaptive Synthesis)Task: Adaptive weighted GPR data fusion.Key Features:Fuses Stage 1 pseudo-results with real high-frequency GPR data.Captures long-range dependencies via Transformer blocks.Outputs high-resolution, high-fidelity fused GPR profiles.📁 Project StructurePlaintextGPR-Fusion-Framework/
├── cycle_gan/              # Stage 1: Domain alignment module
│   ├── datasets/gpr/       # Training/Testing GPR data
│   ├── options/            # Hyperparameter configurations
│   ├── requirements.txt    # Stage 1 dependencies
│   └── README.md           # Detailed Stage 1 guide
├── transformer/            # Stage 2: Fusion module
│   ├── data/               # Demo and fusion input data
│   ├── requirements.txt    # Stage 2 dependencies
│   └── README.md           # Detailed Stage 2 guide
├── LICENSE                 # MIT License
└── README.md               # Main entry point
🚀 Quick Start1. RequirementsPython $\ge$ 3.7PyTorch $\ge$ 1.8CUDA-enabled GPU (Recommended)2. Stage 1: Cycle-GAN Training & InferenceBashcd cycle_gan
pip install -r requirements.txt
# Place your datasets in ./datasets/gpr/
# Configure parameters in ./options/
# Refer to cycle_gan/README.md for execution commands
3. Stage 2: Transformer FusionBashcd ../transformer
pip install -r requirements.txt
# Place Stage 1 outputs into ./data/
# Run the fusion inference (demo data included)
python test_fusion.py  # Example command
📊 Data PreparationModuleData PathDescriptionCycle-GAN./cycle_gan/datasets/gpr/Unpaired L-freq and H-freq GPR B-scansTransformer./transformer/data/Stage 1 results + Real H-freq dataNote: We provide demo data in the transformer/data/ directory. You can run the Stage 2 inference directly to verify the installation and view the fusion performance.📜 CitationIf you find this code or research helpful, please cite our paper:代码段@article{YourName2026GPR,
  title={Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network},
  author={Your Name, Xihong Cui, et al.},
  journal={Computers & Geosciences},
  year={2026}
}
⚖️ LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
