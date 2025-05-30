import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #CUDA_VISIBLE_DEVICES=3
from core.datasets.datasets import *
from core.models.transformer import *
from core.trains.train import *


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default=42, type=int, help='random seeds')
    parser.add_argument('--epochs', default=1000, type=int, help='Num of training epoches')
    parser.add_argument('--ratios', default=100, type=int, help='Ratio of training dataset')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--max_lr', default=1.0, type=float, help='maximum learning rate')
    parser.add_argument('--warm_steps', default=100000, type=int, help='learning rate warmup steps')

    parser.add_argument('--smoothing', default=0.1, type=float, help='Smoothing for KL divergence')

    parser.add_argument('--struc_dim', default=104, type=int, help='Num of struc tokens')
    parser.add_argument('--spec_dim', default=142, type=int, help='Spec dimension')

    parser.add_argument('--layers', default=1, type=int, help='Encoder layers')
    parser.add_argument('--head_num', default=8, type=int, help='Attention head numbers')
    parser.add_argument('--d_model', default=1024, type=int, help='Total attention dim = head_num * head_dim')
    parser.add_argument('--d_ff', default=512, type=int, help='Feed forward layer dim')
    parser.add_argument('--max_len', default=22, type=int, help='Transformer horizons')

    parser.add_argument('--save_folder', default='test', type=str, help='First order folder')
    parser.add_argument('--save_name', default='model_inverse', type=str, help='First order folder')
    parser.add_argument('--spec_type', default='R_T', type=str, help='If predict R/T/R+T')
    parser.add_argument('--TRAIN_FILE', default='TRAIN_FILE', type=str, help='TRAIN_FILE')
    parser.add_argument('--TRAIN_SPEC_FILE', default='TRAIN_SPEC_FILE', type=str, help='TRAIN_SPEC_FILE')
    parser.add_argument('--DEV_FILE', default='DEV_FILE', type=str, help='DEV_FILE')
    parser.add_argument('--DEV_SPEC_FILE', default='DEV_SPEC_FILE', type=str, help='DEV_SPEC_FILE')
    parser.add_argument('--struc_index_dict', default={2:'BOS'}, type=dict, help='struc_index_dict')
    parser.add_argument('--struc_word_dict', default={'BOS':2}, type=dict, help='struc_word_dict')

    args = parser.parse_args()

    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)

    temp = [args.ratios, args.smoothing, args.batch_size, args.max_lr, args.warm_steps, args.layers, args.head_num, args.d_model, args.d_ff]
    args.save_name += '_' + args.spec_type
    args.save_name += '_S_R_B_LR_WU_L_H_D_F_'+str(temp)

    TRAIN_FILE = './dataset/Structure_train.pkl'   
    TRAIN_SPEC_FILE = './dataset/Spectrum_train.pkl'  
    DEV_FILE = './dataset/Structure_dev.pkl'   
    DEV_SPEC_FILE = './dataset/Spectrum_dev.pkl'  

    args.TRAIN_FILE, args.TRAIN_SPEC_FILE, args.DEV_FILE, args.DEV_SPEC_FILE = TRAIN_FILE, TRAIN_SPEC_FILE, DEV_FILE, DEV_SPEC_FILE

    data = PrepareData(TRAIN_FILE, TRAIN_SPEC_FILE, args.ratios, DEV_FILE, DEV_SPEC_FILE, args.batch_size, args.spec_type, 'Inverse')

    tgt_vocab = len(data.struc_word_dict)
    src_vocab = len(data.dev_spec[0])
    args.struc_dim = tgt_vocab
    args.spec_dim = src_vocab
    args.struc_index_dict = data.struc_index_dict
    args.struc_word_dict = data.struc_word_dict

    print(f"struc_vocab {src_vocab}")
    print(f"spec_vocab {tgt_vocab}")

    model = make_model_I(
                    args.spec_dim, 
                    args.struc_dim,
                    args.layers, 
                    args.d_model, 
                    args.d_ff,
                    args.head_num,
                    args.dropout
                ).to(DEVICE)

    print('Model Transformer, Number of parameters {}'.format(count_params(model)))

    # Step 3: Training model
    print(">>>>>>> start train")
    train_start = time.time()
    criterion = LabelSmoothing(tgt_vocab, padding_idx = 0, smoothing= args.smoothing)
    
    optimizer = NoamOpt(args.d_model, args.max_lr, args.warm_steps, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

    train_I(data, model, criterion, optimizer, args, DEVICE)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")