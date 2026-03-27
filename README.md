# Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network

Official PyTorch implementation for our paper **Multi-Scene GPR Data Fusion Based on a Sequential Cycle-GAN and Transformer Network**.

This repository contains a **two-stage sequential deep learning framework** designed for multi-scene Ground Penetrating Radar (GPR) data fusion, which combines Cycle-GAN for cross-domain alignment and Transformer for high-quality data fusion.

---

## 📌 Framework Overview
Our framework consists of **two sequential stages**:

### Stage 1: Sequential Cycle-GAN
- **Task**: Unpaired GPR data translation & alignment
- **Function**:
  - Uses **unpaired training datasets**
  - Achieves **time-domain alignment** and **frequency-domain alignment** between low-frequency and high-frequency GPR data
  - Generates **pseudo high-frequency GPR images**

### Stage 2: Transformer Fusion Network
- **Task**: Adaptive weighted GPR data fusion
- **Function**:
  - Fuses the **pseudo high-frequency images (from Stage 1)** with real high-frequency images
  - Outputs **high-quality, high-resolution fused GPR data**
  - Provides **example demo data** for direct running & testing

---

## 📂 Project Structure
