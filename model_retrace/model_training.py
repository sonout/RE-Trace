import os
import numpy as np
import random

import pickle
import argparse
import torch
from torch.utils.data import DataLoader

from model.dataset import TrainDataset
from model.traj2vec import Traj2VecModel
from utils import load_pretrained
from eval_utils import Dataset
import time

seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)

def main(args):

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print("Load train data...")
    start = time.time()
    train = np.load(os.path.join(args.data_path,"train.npz"))
    train_dataset = TrainDataset(train['src'], train['trg'], train['src_len'], train['trg_len'])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10)
    elapsed_time = time.time() - start
    print(f"Loaded in {elapsed_time} seconds")

    print("Load cell representations...")
    cell_reps = load_pretrained(args.cell_rep_path, args.vocab_size).to(device)
    
    # Init Validation Setup
    print("Load evaluation data...")
    total_set_labels = pickle.load(open(os.path.join(args.data_eval_path, "total_set_labels.pkl"), "rb"))
    own_data_path = os.path.join(args.data_path, "own.t")
    own_dataset = Dataset(own_data_path, args.max_seq_len)
    own_dataloader = DataLoader(own_dataset, batch_size=args.batch_size, shuffle=False)
    total_data_path = os.path.join(args.data_path, "total.t")
    total_dataset = Dataset(total_data_path, args.max_seq_len)
    total_dataloader = DataLoader(total_dataset, batch_size=args.batch_size, shuffle=False)


    print("Train Model...")
    start = time.time()
    model = Traj2VecModel(hidden_dim=args.hidden_dim,emb_dim=args.emb_dim, vocab_size=args.vocab_size, cell_reps=cell_reps, device=device, lr=args.lr, weight_decay=args.weight_decay, T_max=len(train_dataloader), save_model_path =args.save_model_path)
    model.train(args.epochs, train_dataloader, own_dataloader, total_dataloader, total_set_labels)
    model.save_model(args.save_model_path)
    print(f"Saved model at {args.save_model_path}")
    elapsed_time = time.time() - start
    print(f"Training took {elapsed_time} seconds")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='traj2vec')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--data_eval_path', default="/data/shared/schestakov/projects/re-identification/data/porto/db", type=str)
    parser.add_argument('--data_path', default="/data/shared/schestakov/projects/re-identification/data/porto/db/dl_models", type=str)
    parser.add_argument('--save_model_path', default="/data/shared/schestakov/projects/re-identification/model_retrace/states/porto/", type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--vocab_size', default=18866, type=int) # 18866 for porto, 34109 for sf
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--cell_rep_path', default="/home/schestakov/projects/re-identification/model_retrace/cell_rep/porto_128_gridsize_100.txt", type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--emb_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--lr', default=1e-4, type=int) 
    parser.add_argument('--weight_decay', default=1e-4, type=int)

    args = parser.parse_args()

    main(args)

