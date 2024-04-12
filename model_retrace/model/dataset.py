import torch
import numpy as np
from . import transforms as T

class TrainDataset:
    def __init__(self, src, trg, src_len, trg_len):
        # Shape: (N, #views, max_traj_len)
        self.src = src
        self.trg = trg
        # Shape: (N, #views)
        self.src_len = src_len
        self.trg_len = trg_len
        self.length = len(self.trg)
        #self.length = 1284511

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        j = np.random.randint(self.src.shape[1])
        src = torch.tensor(self.src[i,j,:], dtype=torch.long)
        tgt = torch.tensor(self.trg[i,j,:], dtype=torch.long)
        src_len = torch.tensor(self.src_len[i,j], dtype=torch.long)
        trg_len = torch.tensor(self.trg_len[i,j], dtype=torch.long)      

        # with propability of 50% swap src and tgt
        if np.random.rand() > 0.5:
            src, tgt = tgt, src
            src_len, trg_len = trg_len, src_len

        return src, tgt, src_len, trg_len


class ReTraceDataset:
    def __init__(self, src, trg, src_len, trg_len):
        # Shape: (N, #views, max_traj_len)
        self.src = src
        self.trg = trg
        # Shape: (N, #views)
        self.src_len = src_len
        self.trg_len = trg_len
        self.length = len(self.trg)
        #self.length = 1284511

        transform = [
            #T.Simplify(p=0.3),
            #T.Shift(p=0.3),
            T.Mask(p=0.3),
            T.Subset(p=0.3),
        ]
        self.transform = T.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        j = np.random.randint(self.src.shape[1])

        src = self.src[i,j,:]
        trg = self.trg[i,j,:]

        # We need to unpad, transform and then pad again
        max_seq_len = src.shape[0]
        src_unpaded = np.trim_zeros(src)
        src = self.transform(src_unpaded)
        src = np.pad(src, (0, max_seq_len - len(src)), 'constant')

        # Try same for target
        trg_unpaded = np.trim_zeros(trg)
        trg = self.transform(trg_unpaded)
        trg = np.pad(trg, (0, max_seq_len - len(trg)), 'constant')

        src = torch.tensor(src, dtype=torch.long)
        trg = torch.tensor(trg, dtype=torch.long)

        src_len = torch.tensor(self.src_len[i,j], dtype=torch.long)
        trg_len = torch.tensor(self.trg_len[i,j], dtype=torch.long)      
        return src, trg, src_len, trg_len