import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint

class HyperEdgeAttention(nn.Module):
    """Hyper-edge partitioning + MHA   (3-D version)."""
    def __init__(self, channels: int, edges: int, heads: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if channels % heads:
            raise ValueError(f"{channels=} not divisible by {heads=}")
        
        self.edges = edges
        self.in_norm = nn.LayerNorm(channels)
        self.QKV = nn.Linear(channels, channels * 3, bias=bias)
        self.hyperedge = nn.Linear(channels, edges, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.mha = nn.MultiheadAttention(channels, heads,
                                         batch_first=True,
                                         bias=bias,
                                         dropout=dropout)
        self.z_norm = nn.LayerNorm(channels)

    def _compute_edges(self, tokens, scale):
        w = self.hyperedge(tokens)
        w = F.relu(w) * F.softmax(w, dim=-1) * scale
        w = self.dropout(w)
        z = self.z_norm(w.transpose(1, 2) @ tokens)
        return z

    def forward(self, x):
        # x: (B, C, S1, S2, S3)
        B, C, S1, S2, S3 = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).reshape(B, S1 * S2 * S3, C)
        tokens = self.in_norm(tokens)

        scale = self.edges / S1 / S2 / S3
        if self.training and x.requires_grad:
            z = checkpoint.checkpoint(self._compute_edges, tokens, scale, use_reentrant=False)
        else:
            z = self._compute_edges(tokens, scale)
            
        y = self.mha(tokens, z, z, need_weights=False)[0]

        y = y.permute(0, 2, 1).reshape(B, C, S1, S2, S3)
        return y
