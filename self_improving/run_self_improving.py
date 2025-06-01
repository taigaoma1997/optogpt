#!/usr/bin/env python
"""
Clean example script for running the self-improving data augmentation pipeline.
This script demonstrates the full workflow without hardcoded paths.
"""

import argparse
import os
import torch
import pandas as pd
from pathlib import Path

from prepare_aug_data import Prepare_Augment_Data
from data_perturb import get_to_be_perturbed_data, perturb_data, simulate_perturbed_struct, get_perturbed_better_data
from combine_data import combine_data
from model_retrain import retrain_model
from generate_dev_data import generate_dev_data
from core.datasets.datasets import PrepareData


def parse_args():
    parser = argparse.ArgumentParser(description='Self-Improving Data Augmentation for OptoGPT')
    
    # Required paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to pretrained OptoGPT model')
    parser.add_argument('--train_struct_path', type=str, required=True,
                        help='Path to training structure data (pkl file)')
    parser.add_argument('--train_spec_path', type=str, required=True,
                        help='Path to training spectrum data (pkl file)')
    parser.add_argument('--dev_struct_path', type=str, required=True,
                        help='Path to development structure data (pkl file)')
    parser.add_argument('--dev_spec_path', type=str, required=True,
                        help='Path to development spectrum data (pkl file)')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory to save outputs')
    
    # Method configurations
    parser.add_argument('--decoding_method', type=str, default='TOP-KP Decode_v2',
                        choices=['Greedy Decode', 'TOP-KP Decode', 'TOP-KP Decode_v2', 'Beam Search'],
                        help='Decoding method for data generation')
    parser.add_argument('--perturbation_method', type=str, default='GA_PSO',
                        choices=['random', 'PSO', 'GA_PSO'],
                        help='Perturbation method')
    parser.add_argument('--error_type', type=str, default='MSE',
                        choices=['MAE', 'MSE'],
                        help='Error metric type')
    
    # Hyperparameters
    parser.add_argument('--give_up_threshold', type=float, default=3.0,
                        help='Maximum error threshold for structures')
    parser.add_argument('--kp_num', type=int, default=50,
                        help='Number of decoding attempts per spectrum')
    parser.add_argument('--keep_num', type=int, default=20,
                        help='Number of best structures to keep')
    parser.add_argument('--target_aug_size', type=int, default=200000,
                        help='Target size for augmented dataset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of retraining epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of CPU workers for multiprocessing')
    
    return parser.parse_args()


def setup_directories(output_dir):
    """Create necessary output directories."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'augmented_data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)


def main():
    args = parse_args()
    
    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # Override hardcoded devices in modules
    import prepare_aug_data
    import data_perturb
    prepare_aug_data.DEVICE = device
    data_perturb.DEVICE = device
    
    # Set multiprocessing workers
    import multiprocessing
    if args.num_workers > 0:
        multiprocessing.set_start_method('spawn', force=True)
    
    # Setup directories
    setup_directories(args.output_dir)
    
    print(f"Running self-improving data augmentation...")
    print(f"Device: {device}")
    print(f"Output directory: {args.output_dir}")
    
    # Step 1: Generate development data
    print("\n1. Generating development data...")
    dev_spec = generate_dev_data()
    print(f"Generated {len(dev_spec)} development spectra")
    
    # Step 2: Prepare augmentation data
    print("\n2. Preparing augmentation data...")
    augment_data = Prepare_Augment_Data(
        model_path=args.model_path,
        decoding_method=args.decoding_method,
        error_type=args.error_type,
        top_k=10,
        top_p=0.9,
        kp_num=args.kp_num,
        keep_num=args.keep_num
    )
    
    # Step 3: Get data to be perturbed
    print("\n3. Selecting data for perturbation...")
    perturbed_df = get_to_be_perturbed_data(augment_data, args.give_up_threshold)
    print(f"Selected {len(perturbed_df)} structures for perturbation")
    
    # Step 4: Perturb data
    print(f"\n4. Perturbing data using {args.perturbation_method}...")
    perturbed_df = perturb_data(perturbed_df, method=args.perturbation_method)
    
    # Save intermediate results
    perturbed_df.to_pickle(os.path.join(args.output_dir, 'augmented_data', 'perturbed_data.pkl'))
    
    # Step 5: Simulate perturbed structures
    print("\n5. Simulating perturbed structures...")
    perturbed_df = simulate_perturbed_struct(perturbed_df, error_type=args.error_type)
    
    # Step 6: Filter better data
    print("\n6. Filtering improved structures...")
    added_data = get_perturbed_better_data(perturbed_df)
    initial_size = len(added_data)
    print(f"Found {initial_size} improved structures")
    
    # Remove duplicates
    added_data = added_data.drop_duplicates(subset=['new_error'])
    
    # Duplicate to reach target size
    while len(added_data) < args.target_aug_size:
        added_data = pd.concat([added_data, added_data], ignore_index=True)
    added_data = added_data[:args.target_aug_size]
    
    # Save augmented data
    added_data.to_pickle(os.path.join(args.output_dir, 'augmented_data', 'added_data.pkl'))
    print(f"Augmented dataset size: {len(added_data)}")
    
    # Step 7: Combine data
    print("\n7. Combining original and augmented data...")
    new_train_spec_path, new_train_struct_path, new_test_spec_path, new_test_struct_path = combine_data(
        args.train_spec_path,
        args.train_struct_path,
        args.dev_spec_path,
        args.dev_struct_path,
        added_data,
        ratio=0.1,
        type_T=args.decoding_method
    )
    
    # Move combined data to output directory
    import shutil
    for old_path, new_name in [
        (new_train_spec_path, 'train_spec_augmented.pkl'),
        (new_train_struct_path, 'train_struct_augmented.pkl'),
        (new_test_spec_path, 'test_spec_augmented.pkl'),
        (new_test_struct_path, 'test_struct_augmented.pkl')
    ]:
        new_path = os.path.join(args.output_dir, 'augmented_data', new_name)
        shutil.move(old_path, new_path)
        if 'train_spec' in new_name:
            new_train_spec_path = new_path
        elif 'train_struct' in new_name:
            new_train_struct_path = new_path
        elif 'test_spec' in new_name:
            new_test_spec_path = new_path
        elif 'test_struct' in new_name:
            new_test_struct_path = new_path
    
    # Step 8: Load original model configuration
    print("\n8. Loading model configuration...")
    model_checkpoint = torch.load(args.model_path, map_location=device)
    original_args = model_checkpoint['configs']
    
    # Get vocabulary from original data
    data = PrepareData(
        args.train_struct_path, 
        args.train_spec_path, 
        original_args.ratios, 
        args.dev_struct_path, 
        args.dev_spec_path, 
        original_args.batch_size, 
        original_args.spec_type, 
        'Inverse'
    )
    struct_word_dict = data.struc_word_dict
    struct_index_dict = data.struc_index_dict
    
    # Step 9: Retrain model
    print(f"\n9. Retraining model for {args.epochs} epochs...")
    
    # Override device selection in retrain function
    import model_retrain
    model_retrain.DEVICE = device
    
    # Models directory
    models_dir = os.path.join(args.output_dir, 'models')
    
    retrain_model(
        args.model_path,
        new_train_struct_path,
        new_train_spec_path,
        new_test_spec_path,
        new_test_struct_path,
        args.epochs,
        args.early_stopping_patience,
        struct_word_dict,
        struct_index_dict,
        args.decoding_method,
        device=device,
        output_dir=models_dir
    )
    
    print(f"\n✓ Self-improving data augmentation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"- Augmented data: {os.path.join(args.output_dir, 'augmented_data')}")
    print(f"- Trained models: {models_dir}")


if __name__ == '__main__':
    main() 