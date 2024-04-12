import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@dataclass
class ModelArgs:
    dim: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    hidden_dim: Optional[int] = 2048
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 100
    dropout: float = 0.1


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        else:
            print("Warning: I did not check manual implementation!!! Mask is probably off")
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim) # (B, n_heads, seqlen, head_dim) @ (B, n_heads, head_dim, seqlen) ---> (B, n_heads, seqlen, seqlen)
            if attn_mask is not None: # This version supports only boolean mask
                # True indicates that the element should take part in attention.
                scores = scores.masked_fill(attn_mask == False, float('-inf'))
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin, mask: Optional[torch.Tensor] = None):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Llama_tranformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)


        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
        
        self.lstm_agg = nn.LSTM(
            input_size=params.dim,
            hidden_size=params.dim,
            num_layers=1,
            batch_first=True,
        )


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = x # shape: [batch_size, max_seq_len, emb_dim]
        _bsz, seqlen, _emb_dim = h.shape
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Our masking
        max_src_len = torch.arange(x.shape[1]).to(x.get_device()) # in essense -- trajs1_len[0]
        src_padding_mask = max_src_len[None, :] < lengths[:, None]
        src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.params.n_heads, -1, -1) # Do not know if I should do that or get a mask without heads_dim
        # Maybe even : src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.params.n_heads, seqlen, -1) ? So that I have a seqlen X seqlen mask?
        #src_padding_mask = src_padding_mask.view(_bsz, 1, seqlen) # Maybe even

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, src_padding_mask)
        h = self.norm(h)

        mask = max_src_len[None, :] >= lengths[:, None]
        mask = 1 - mask.unsqueeze(-1).expand(h.shape).float()
        h = torch.sum(mask * h, 1) # Sum over all seq_len
        h = h / lengths.unsqueeze(-1).expand(h.shape) # Normalize by lengths of traj

        ## LSTM 
        # self.lstm_agg.flatten_parameters()
        # outputs, (hs, cs) = self.lstm_agg(h) # Outputs shape: same as rtn: [batch_size, max_seq_len, emb_size]
        # h = outputs[torch.arange(h.shape[0]), lengths-1]

        return h







class AttentionAgg(nn.Module):
    def __init__(self, emb_dim):
        super(AttentionAgg, self).__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, 1) # try bias = False

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, emb_dim)
        # mask shape: (batch_size, seq_len)
        attn_weights = self.linear(x).squeeze(-1)  # shape: (batch_size, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)  # apply mask
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
        output = (x * attn_weights).sum(dim=1)  # shape: (batch_size, emb_dim)
        return output


class AttentionAgg2(nn.Module):
    def __init__(self, emb_dim, seq_len=100):
        super(AttentionAgg2, self).__init__()
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, 1) # try bias = False

        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)

        self.w_m = nn.Linear(seq_len, seq_len, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, emb_dim)
        # mask shape: (batch_size, seq_len)
        epsilon = 1e-7

        # The temporal dif between each token is the same
        # Create a tensor with shape [batch_size, seq_len, seq_len] filled with tange from 0 to seq_len
        range_seq = torch.arange(x.shape[1]).to(x.get_device()) # in essense -- trajs1_len[0]
        delta = (range_seq[:, None] - range_seq[None, :]).abs().float() + epsilon
        delta = self.relu(self.w_m(delta))

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        ## Attention
        #x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        wei =  q @ k.transpose(-2, -1) # (B, T, d) @ (B, d, T) ---> (B, T, T)
        wei = wei + 1/(torch.log((delta + epsilon)+epsilon)) # (B, T, T) + (T, T) ---> (B, T, T)
        wei = wei.masked_fill(mask.view(mask.shape[0], 1, mask.shape[-1]) == 0, -1e9)
        wei = F.softmax(wei, dim=-1)
        out = wei @ v # (B, T, T) @ (B, T, d) ---> (B, T, d)

        attn_weights = self.linear(out).squeeze(-1)  # shape: (batch_size, seq_len)
        attn_weights = attn_weights.masked_fill(mask == 0, -1e9)  # apply mask
        attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
        output = (out * attn_weights).sum(dim=1)  # shape: (batch_size, emb_dim)
        return output
    

    class AttentionAgg3(nn.Module):
        def __init__(self, emb_dim, seq_len=100):
            super(AttentionAgg2, self).__init__()
            self.emb_dim = emb_dim
            self.linear = nn.Linear(emb_dim, 1) # try bias = False

            self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
            self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
            self.wv = nn.Linear(emb_dim, emb_dim, bias=False)

            self.w_m = nn.Linear(seq_len, seq_len, bias=True)

        def forward(self, x, mask):
            # x shape: (batch_size, seq_len, emb_dim)
            # mask shape: (batch_size, seq_len)

            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)

            ## Attention
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

            attn_weights = self.linear(x).squeeze(-1)  # shape: (batch_size, seq_len)
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)  # apply mask
            attn_weights = F.softmax(attn_weights, dim=-1).unsqueeze(-1)  # shape: (batch_size, seq_len, 1)
            output = (x * attn_weights).sum(dim=1)  # shape: (batch_size, emb_dim)
            return output



class Llama_tranformer_Att(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)


        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
        
        self.att_agg = AttentionAgg2(params.dim)
        self.norm2 = RMSNorm(params.dim, eps=params.norm_eps)
        self.linear = nn.Linear(params.dim, params.dim)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = x # shape: [batch_size, max_seq_len, emb_dim]
        _bsz, seqlen, _emb_dim = h.shape
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Our masking
        max_src_len = torch.arange(x.shape[1]).to(x.get_device()) # in essense -- trajs1_len[0]
        src_padding_mask = max_src_len[None, :] < lengths[:, None]
        src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.params.n_heads, -1, -1) # Do not know if I should do that or get a mask without heads_dim
        # Maybe even : src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.params.n_heads, seqlen, -1) ? So that I have a seqlen X seqlen mask?
        #src_padding_mask = src_padding_mask.view(_bsz, 1, seqlen) # Maybe even

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, src_padding_mask)
        h = self.norm(h)

        mask = max_src_len[None, :] < lengths[:, None]
        h = self.norm2(h)
        h = self.att_agg(h, mask)
        # h = self.norm2(h)
        # h = self.linear(h)

        # mask = max_src_len[None, :] >= lengths[:, None]
        # mask = 1 - mask.unsqueeze(-1).expand(h.shape).float()
        # h = torch.sum(mask * h, 1) # Sum over all seq_len
        # h = h / lengths.unsqueeze(-1).expand(h.shape) # Normalize by lengths of traj

        return h







from lightly.models.modules import NNCLRPredictionHead
class Llama_tranformer_lstm(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.n_layers = params.n_layers

        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)


        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))
        
        self.lstm = nn.LSTM(
            input_size=params.dim,
            hidden_size=params.dim,
            num_layers=1,
            batch_first=True,
        )
        self.linear = NNCLRPredictionHead(params.dim, 2048, params.dim)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        #### LSTM ###
        self.lstm.flatten_parameters()
        output, (h_n, c_n) = self.lstm(x)
        hn_lstm = output[torch.arange(x.shape[0]), lengths-1]


        h = x # shape: [batch_size, max_seq_len, emb_dim]
        _bsz, seqlen, _emb_dim = h.shape
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Our masking
        max_src_len = torch.arange(x.shape[1]).to(x.get_device()) # in essense -- trajs1_len[0]
        src_padding_mask = max_src_len[None, :] < lengths[:, None]
        src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.params.n_heads, -1, -1) # Do not know if I should do that or get a mask without heads_dim
        # Maybe even : src_padding_mask = src_padding_mask.view(_bsz, 1, 1, seqlen).expand(-1, self.params.n_heads, seqlen, -1) ? So that I have a seqlen X seqlen mask?
        #src_padding_mask = src_padding_mask.view(_bsz, 1, seqlen) # Maybe even

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin, src_padding_mask)
        h = self.norm(h)

        mask = max_src_len[None, :] >= lengths[:, None]
        mask = 1 - mask.unsqueeze(-1).expand(h.shape).float()
        h = torch.sum(mask * h, 1) # Sum over all seq_len
        h = h / lengths.unsqueeze(-1).expand(h.shape) # Normalize by lengths of traj


        # Concatenate hn_lstm and hn_att
        #hn = torch.cat((hn_lstm, hn_att), dim=1)
        h = hn_lstm + h
        h = self.linear(h)

        ## LSTM 
        
        # outputs, (hs, cs) = self.lstm_agg(h) # Outputs shape: same as rtn: [batch_size, max_seq_len, emb_size]
        # h = outputs[torch.arange(h.shape[0]), lengths-1]
        return h
    