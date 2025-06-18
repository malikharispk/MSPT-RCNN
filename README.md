# MSPT-RCNN Implementation for 3D Object Detection

This repository contains the implementation of MSPT-RCNN for 3D object detection using the KITTI dataset.

## Overview

MSPT-RCNN is designed for 3D object detection in autonomous driving scenarios. The model leverages:
- **Neighborhood Embedding Module**: Captures local geometric features.
- **Offset Attention Mechanism**: A multi-scale attention layer for feature propagation.
- **Region Proposal Network (RPN)**: Proposes initial bounding boxes for objects.
- **RCNN**: Refines the bounding boxes using detailed local spatial features.

## Dataset

We use the **KITTI dataset** for 3D object detection. You can download the dataset from the [KITTI website](http://www.cvlibs.net/datasets/kitti/).

Ensure that you preprocess the KITTI data and store it in the format specified in the `config.yaml` file.

## Requirements

- Python 3.7
- PyTorch
- NumPy
- Open3D
- Matplotlib
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
