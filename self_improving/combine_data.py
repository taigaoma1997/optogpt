import sys
sys.path.append('../optogpt')
import os
import numpy as np
import pickle as pkl

def combine_data(original_spec_train, original_struct_train, original_spec_test, original_struct_test, 
                 new_data, ratio=0.1, type_T="Greedy Decode", output_dir='./dataset'):
    """
    Separate the new data into train and test data, and combine the original data and new data
    
    Args:
        original_spec_train: Path to original training spectrum data
        original_struct_train: Path to original training structure data
        original_spec_test: Path to original test spectrum data
        original_struct_test: Path to original test structure data
        new_data: DataFrame with augmented data
        ratio: Ratio for train/test split
        type_T: Type of decoding method used
        output_dir: Directory to save output files
    
    Returns:
        Tuple of paths to new train/test spec/struct files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(original_struct_train, 'rb') as fp:
        train_struc = pkl.load(fp)

    with open(original_spec_train, 'rb') as fp:
        train_spec = pkl.load(fp)       

    with open(original_struct_test, 'rb') as fp:
        test_struc = pkl.load(fp)
    
    with open(original_spec_test, 'rb') as fp:
        test_spec = pkl.load(fp)

    print("Read original data")
    
    # Separate the new data into train and test data, size 100:1
    new_data = new_data.sample(frac=1).reset_index(drop=True)
    new_data_train = new_data[:int(len(new_data)*0.99)]
    new_data_test = new_data[int(len(new_data)*0.99):]
    
    # Combine the original data and new data
    add_struct_train = new_data_train["perturb_struct"].tolist()
    add_spec_train = new_data_train["perturb_spec"].tolist()

    add_struct_test = new_data_test["perturb_struct"].tolist()
    add_spec_test = new_data_test["perturb_spec"].tolist()

    new_train_struct = train_struc + add_struct_train
    new_train_spec = np.array(train_spec.tolist() + add_spec_train)

    new_test_struct = test_struc + add_struct_test
    new_test_spec = np.array(test_spec.tolist() + add_spec_test)

    print("Combined data")
    
    # Convert decoding type to simplified string for filenames
    if type_T == "Greedy Decode":
        method_suffix = "greedy"
    elif type_T == "TOP-KP Decode":
        method_suffix = "topkp"
    elif type_T == "TOP-KP Decode_v2":
        method_suffix = "topkp_v2"
    elif type_T == "Beam Search":
        method_suffix = "beam"
    else:
        method_suffix = "custom"
    
    # Create consistent naming pattern
    train_spec_filename = f"train_spectrum_augmented_{method_suffix}.pkl"
    train_struct_filename = f"train_structure_augmented_{method_suffix}.pkl"
    test_spec_filename = f"test_spectrum_augmented_{method_suffix}.pkl"
    test_struct_filename = f"test_structure_augmented_{method_suffix}.pkl"
    
    # Full paths
    train_spec_path = os.path.join(output_dir, train_spec_filename)
    train_struct_path = os.path.join(output_dir, train_struct_filename)
    test_spec_path = os.path.join(output_dir, test_spec_filename)
    test_struct_path = os.path.join(output_dir, test_struct_filename)
    
    # Save the combined data
    with open(train_spec_path, 'wb') as fp:
        pkl.dump(new_train_spec, fp)
    
    with open(train_struct_path, 'wb') as fp:
        pkl.dump(new_train_struct, fp)

    with open(test_spec_path, 'wb') as fp:
        pkl.dump(new_test_spec, fp)

    with open(test_struct_path, 'wb') as fp:
        pkl.dump(new_test_struct, fp)
    
    print(f"Saved combined data with {method_suffix} method:")
    print(f"  Train files: {train_spec_filename}, {train_struct_filename}")
    print(f"  Test files: {test_spec_filename}, {test_struct_filename}")
        
    return train_spec_path, train_struct_path, test_spec_path, test_struct_path