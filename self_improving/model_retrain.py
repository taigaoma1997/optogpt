import sys
sys.path.append('../optogpt')
import os
import time
import torch
import numpy as np
from core.datasets.datasets import PrepareDataAug
from core.models.transformer import make_model_I
from core.trains.train import LabelSmoothing, run_epoch_I, SimpleLossCompute

# Default device can be overridden from outside
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_all,
        'configs': configs
    }, path)

def train_I_aug(data, model, criterion, optimizer, configs, device, epochs, early_stopping_patience, type_T, output_dir="./saved_models"):
    """
    Train and Save the model with early stopping.
    
    Args:
        data: Dataset object containing train and dev data
        model: Model to train
        criterion: Loss criterion
        optimizer: Optimizer
        configs: Configuration parameters
        device: Device to train on
        epochs: Number of epochs to train
        early_stopping_patience: Early stopping patience
        type_T: Type of decoding method used
        output_dir: Directory to save model checkpoints
    """
    best_dev_loss = 1e5
    loss_all = {'train_loss': [], 'dev_loss': []}
    save_name = configs.save_name if hasattr(configs, 'save_name') else 'model'
    EPOCHS = epochs
    epochs_without_improvement = 0
    
    # Set up model save directory based on decoding type
    if type_T == "Greedy Decode":
        save_dir = os.path.join(output_dir, "greedy_decode")
    elif type_T == "TOP-KP Decode":
        save_dir = os.path.join(output_dir, "topkp_decode") 
    elif type_T == "TOP-KP Decode_v2":
        save_dir = os.path.join(output_dir, "topkp_decode_v2")
    elif type_T == "Beam Search":
        save_dir = os.path.join(output_dir, "beam_search")
    else:
        save_dir = os.path.join(output_dir, "custom")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = os.path.join(save_dir, f"{save_name}_best_augmented.pt")
    recent_model_path = os.path.join(save_dir, f"{save_name}_recent_augmented.pt")

    print(f"Models will be saved to: {save_dir}")

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # Train model
        model.train()
        train_loss = run_epoch_I(data.train_data, model, SimpleLossCompute(model.generator, criterion, optimizer), epoch, device)
        model.eval()

        # validate model on dev dataset
        print('>>>>> Evaluate')
        dev_loss = run_epoch_I(data.dev_data, model, SimpleLossCompute(model.generator, criterion, None), epoch, device)
        print(f'<<<<< Evaluate loss: {dev_loss:.2f} (Epoch {epoch+1}/{EPOCHS}, {time.time()-start_time:.2f}s)')
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())

        # Check for early stopping
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            epochs_without_improvement = 0
            # Save the model if it's the best so far
            save_checkpoint(model, optimizer, epoch, loss_all, best_model_path, configs)
            print(f"New best model saved (loss: {dev_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break

        # Save the recent model
        save_checkpoint(model, optimizer, epoch, loss_all, recent_model_path, configs)

class FinetuneOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor
     
def retrain_model(model_path, new_train_struct, new_train_spec, new_test_struct, new_test_spec, 
                  epochs, early_stopping_patience, struct_word_dict, struct_index_dict, type_T, device=None, output_dir="./saved_models"):
    """
    Retrain the model with augmented data.
    
    Args:
        model_path: Path to the pretrained model
        new_train_struct: Path to augmented training structure data
        new_train_spec: Path to augmented training spectrum data
        new_test_struct: Path to augmented test structure data
        new_test_spec: Path to augmented test spectrum data
        epochs: Number of training epochs
        early_stopping_patience: Early stopping patience
        struct_word_dict: Structure word dictionary
        struct_index_dict: Structure index dictionary
        type_T: Type of decoding method used
        device: Device to use (if None, uses global DEVICE)
        output_dir: Directory to save model outputs
    """
    if device is None:
        device = DEVICE
    
    a = torch.load(model_path, map_location=device)
    args = a['configs']
    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)
    model = make_model_I(
                    args.spec_dim, 
                    args.struc_dim,
                    args.layers, 
                    args.d_model, 
                    args.d_ff,
                    args.head_num,
                    args.dropout
                ).to(device)

    model.load_state_dict(a['model_state_dict'])

    TRAIN_FILE = new_train_struct
    TRAIN_SPEC_FILE = new_train_spec
    DEV_FILE = new_test_struct
    DEV_SPEC_FILE = new_test_spec

    args.TRAIN_FILE, args.TRAIN_SPEC_FILE, args.DEV_FILE, args.DEV_SPEC_FILE = TRAIN_FILE, TRAIN_SPEC_FILE, DEV_FILE, DEV_SPEC_FILE

    data = PrepareDataAug(TRAIN_FILE, TRAIN_SPEC_FILE, args.ratios, DEV_FILE, DEV_SPEC_FILE, args.batch_size, args.spec_type, 'Inverse', struct_word_dict, struct_index_dict)
    tgt_vocab = len(data.struc_word_dict)
    print(tgt_vocab)
    src_vocab = len(data.dev_spec[0])
    args.struc_dim = tgt_vocab
    args.spec_dim = src_vocab
    args.struc_index_dict = data.struc_index_dict
    args.struc_word_dict = data.struc_word_dict

    print(">>>>>>> start train")
    criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= args.smoothing)
    
    optimizer = FinetuneOpt(args.d_model, 5e-5, 1, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
 
    train_I_aug(data, model, criterion, optimizer, args, device, epochs, early_stopping_patience, type_T, output_dir)
    print("<<<<<<< finished train")
    

