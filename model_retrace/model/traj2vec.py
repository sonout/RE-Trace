import time
import os
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import accuracy_score, f1_score

import eval_utils
from .traj_encoder_llama import ModelArgs, Llama_tranformer_Att
from .contrastive_frameworks import SimCLR

class Traj2VecModel(object):

    def __init__(self, hidden_dim, emb_dim, vocab_size, device, cell_reps=None,  lr=0.0001, weight_decay=1e-4, T_max=100000, save_model_path="./saved_model"):
        super().__init__()

        self.model = Traj2Vec(
            vocab_size = vocab_size,
            hidden_size = hidden_dim,
            bidirectional = False,
            n_layers = 1,
            emb_dim = emb_dim,
            device = device
        ).to(device)
        if cell_reps is not None:
            self.model.load_pretrained_embedding(cell_reps, freeze=True)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=T_max, eta_min=0, last_epoch=-1)
        self.save_model_path = save_model_path

    def train(self, epochs, train_loader, own_dataloader=None, total_dataloader=None, total_set_labels=None):
        n_iter = 0
        for epoch_counter in range(epochs):
            self.model.train()
            for batch in tqdm(train_loader):

                X1, X2, len1, len2 = batch
                # X1/X2: (batch_size, padded_length, feat_dim)
                # padding_masks: (batch_size, padded_length)
                # batch_temporal_mat: (batch_size, padded_length, padded_length)
                X1 = X1.to(self.device)
                X2 = X2.to(self.device)
                len1 = len1.to(self.device)
                len2 = len2.to(self.device)

                loss = self.model(X1, len1, X2, len2)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                n_iter += 1
            
            f1_score_val, acc_val = self.validate(own_dataloader, total_dataloader, total_set_labels)
            print(f"Epoch: {epoch_counter}\tLoss: {loss}\t Eval F1: {f1_score_val}\t Eval Acc: {acc_val}")
            with open(os.path.join(self.save_model_path, "log.txt"), "a") as f:
                f.write(f"Epoch: {epoch_counter}\tLoss: {loss}\t Eval F1: {f1_score_val}\t Eval Acc: {acc_val} \n")

            # save model 
            self.save_model( path=self.save_model_path, filename=f"model_{epoch_counter}_{f1_score_val:.3f}.pt")

    
    def validate(self, own_dataloader, total_dataloader, total_set_labels):
        # Get embeddings
        own_s_list = []
        total_s_list = []
        with torch.no_grad():
            self.model.eval()
            for batch in own_dataloader:
                X, X_len = batch
                X = X.to(self.device)
                X_len = X_len.to(self.device)
                own_s = self.model.embed_trajectory(X, X_len)
                own_s_list.append(own_s.cpu())
            own_s_list = torch.cat(own_s_list, dim=0).numpy()
            for batch in total_dataloader:
                X, X_len = batch
                X = X.to(self.device)
                X_len = X_len.to(self.device)
                total_s = self.model.embed_trajectory(X, X_len)
                total_s_list.append(total_s.cpu())
            total_s_list = torch.cat(total_s_list, dim=0).numpy()

        # Evaluate
        distances, train_time = eval_utils.execute_test_loop_k(own_s_list, total_s_list, k=1)
        train_split = 0.7
        num_train_samples = int( len(total_s_list) * train_split )
        predictions = eval_utils.train_predict(distances[:num_train_samples], distances[num_train_samples:], total_set_labels[:num_train_samples])
        f1 = f1_score(total_set_labels[num_train_samples:], predictions)
        accuracy = accuracy_score(total_set_labels[num_train_samples:],predictions)
        return f1, accuracy

    def save_model(self, path="./saved_model", filename="model.pt"):
        full_path = os.path.join(path,filename)
        print(f"Model saved to {full_path}")
        torch.save(self.model.state_dict(), full_path)

    def load_model(self, path="./saved_model", filename="model.pt"):
        full_path = os.path.join(path,filename)
        self.model.load_state_dict(torch.load(full_path, map_location=self.device))

    def save_emb(self, path):
        np.savetxt(path + "embedding.out", X=self.model().detach().cpu().numpy())

    def load_emb(self, path=None):
        if path:
            self.emb = np.loadtxt(path)
        return self.model().detach().cpu().numpy()



class Traj2Vec(nn.Module):

    def __init__(
        self, 
        vocab_size, 
        hidden_size, 
        bidirectional, 
        n_layers,
        device,
        emb_dim = 128,
        temp = 0.07
        ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional = True if bidirectional else False
        self.n_direction = 2 if self.bidirectional else 1
        self.n_layers = n_layers

        # TODO: TRy different, and use different
        self.embedding_layer = nn.Embedding(self.vocab_size, emb_dim)
        # the embedding layer is pretrained to 128 dim, so we need to project it to the hidden size
        #self.linear = nn.Linear(128, self.hidden_size)

        args = ModelArgs()
        args.dim = emb_dim
        encoder = Llama_tranformer_Att(args)
        #encoder = TransformerEncoder(emb_dim)
        #encoder = LSTMAttentionEncoder3(self.hidden_size, self.hidden_size, self.bidirectional, self.n_layers, emb_dim)
        #encoder = TransformerEncoder2(emb_dim)
        proj_dim = emb_dim
        self.clmodel = SimCLR(encoder,  emb_dim, proj_dim)
        #self.clmodel = BYOL(encoder,  emb_dim, proj_dim)
        moco_temperature = 0.07
        moco_nqueue=2048
        moco_proj_dim = emb_dim // 2
        # self.clmodel = SimCLR_mmt(encoder,  emb_dim, proj_dim, temperature = moco_temperature)
        # self.clmodel = MoCo(encoder,  
        #                 emb_dim,
        #                 moco_proj_dim, 
        #                 moco_nqueue,
        #                 temperature = moco_temperature)
        

    def load_pretrained_embedding(self, cell_reps, freeze=True):
        self.embedding_layer = self.embedding_layer.from_pretrained(cell_reps, freeze=freeze)

    def reset_parameters(self):
        self.embedding_layer.reset_parameters()
    
    def forward(self, x1, len1, x2, len2):
        # Input: (#B, seq_len)
        # (#B, seq_len, emb_dim)
        x1 = self.embedding_layer(x1)
        x2 = self.embedding_layer(x2)

        loss = self.clmodel({'x': x1, 'lengths': len1},{ 'x': x2, 'lengths': len2})  
        return loss

    def embed_trajectory(self, x, len):
        x = self.embedding_layer(x)
        #x = self.linear(x)
        z = self.clmodel.backbone(**{'x': x, 'lengths': len})
        #z = self.projection_head(z)
        return z





