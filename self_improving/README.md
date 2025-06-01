# Self-Improving Data Augmentation for OptoGPT

This module implements self-improving data augmentation to solve out-of-distribution challenges in optical design foundation models.

## Overview

The self-improving approach automatically generates better training data by:
1. Exploring diverse solutions with multiple decoding strategies
2. Applying intelligent perturbations to promising structures
3. Filtering perturbed structures that improve performance
4. Retraining the model with the augmented dataset

## Quick Start

```bash
python run_self_improving.py \
    --model_path path/to/pretrained_model.pt \
    --train_struct_path path/to/train_struct.pkl \
    --train_spec_path path/to/train_spec.pkl \
    --dev_struct_path path/to/dev_struct.pkl \
    --dev_spec_path path/to/dev_spec.pkl \
    --output_dir ./output \
    --decoding_method TOP-KP_Decode_v2 \
    --perturbation_method GA_PSO \
    --epochs 10
```

## Module Components

### Core Scripts

- **`run_self_improving.py`**: Main pipeline that orchestrates the complete workflow
  
- **`prepare_aug_data.py`**: Prepares augmentation data using different decoding strategies
  - Greedy Decode: Deterministic decoding
  - TOP-KP Decode: Top-k/Top-p sampling for diversity
  - Beam Search: Multi-path exploration

- **`data_perturb.py`**: Implements perturbation strategies
  - Random: Simple thickness perturbation
  - PSO: Particle Swarm Optimization for thickness
  - GA_PSO: Combined Genetic Algorithm and PSO approach

- **`generate_dev_data.py`**: Generates out-of-distribution test spectra
  - Gaussian and Double Gaussian spectra
  - Smooth pulse functions
  - DBR (Distributed Bragg Reflector) structures

- **`combine_data.py`**: Combines original and augmented data

- **`model_retrain.py`**: Fine-tunes model on augmented dataset

## Material Categories

Materials are categorized by refractive index for intelligent perturbation:
- **High Index**: TiO2, ZnS, ZnSe, Ta2O5, HfO2
- **Medium Index**: SiO2, Al2O3, MgF2, Si3N4
- **Low Index**: MgO, ITO
- **Metals**: Al, Ag
- **Semiconductors**: Ge, Si

## Key Parameters

- **Decoding Method**: `Greedy Decode`, `TOP-KP Decode`, `TOP-KP Decode_v2`, `Beam Search`
- **Perturbation Method**: `random`, `PSO`, `GA_PSO`
- **Error Type**: `MAE`, `MSE`

## Requirements

- PyTorch >= 1.8.0
- NumPy
- Pandas
- pyswarms (for PSO optimization)
- scipy
- multiprocessing support

## Notes

- The module uses multiprocessing for efficient parallel computation
- GPU is recommended for model inference and training
- Ensure sufficient disk space for saving augmented datasets
- The `nk` folder with material refractive index data must be accessible 

## Citation

If you use this code, please cite:
```
@inproceedings{ma2024solving,
  title={Solving Out-of-Distribution Challenges in Optical Foundation Models using Self-Improving Data Augmentation},
  author={Ma, Mingqian and Ma, Taigao and Guo, L Jay},
  booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges}
}
```