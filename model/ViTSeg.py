import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import stochastic_depth

class SwiGLU(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_c, h_c * 2, bias)
        self.act = nn.SiLU()
        self.linear2 = nn.Sequential(
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(h_c, out_c, bias))
        
    def forward(self, x):
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        x = self.linear2(self.act(x1) * x2)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, in_c: int, head_dim: int, repeats: int, bias: bool = True,
                 dropout: float = 0.0, sto_depth: float = 0.0):
        super().__init__()
        assert in_c % head_dim == 0, "in_c must be divisible by head_dim"
        self.sto_depth = sto_depth
        self.repeats = repeats
        self.mha_norms = nn.ModuleList([
            nn.LayerNorm(in_c) for _ in range(repeats)])
        self.MHAs = nn.ModuleList([
            nn.MultiheadAttention(in_c, in_c//head_dim, dropout=dropout, 
                        batch_first=True, bias=bias)
            for _ in range(repeats)])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(in_c),
                SwiGLU(in_c, in_c*2, in_c, bias=bias, dropout=dropout))
            for _ in range(repeats)])

    def forward(self, x):
        for norm, mha, mlp in zip(self.mha_norms, self.MHAs, self.mlps):
            norm_x = norm(x)
            x = x + stochastic_depth(mha(norm_x, norm_x, norm_x, need_weights=False)[0], 
                                     self.sto_depth, 'row', self.training)
            x = x + stochastic_depth(mlp(x), self.sto_depth, 'row', self.training)
        return x
    

def get_1d_sinusoidal_pos_embed(length: int, dim: int, device: torch.device):
    """
    Generate a 1D sinusoidal positional embedding table of shape (length, dim).
    """
    # each pair of channels uses one sine and one cosine, so dim must be even
    if dim % 2 != 0:
        raise ValueError("Dimension for sinusoidal embed must be even.")
    position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)  # (L,1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float, device=device)
        * -(torch.log(torch.tensor(10000.0)) / dim)
    )  # (dim/2,)
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class ViTSeg(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        self.in_c = p['in_channels']
        self.out_c = p['out_channels']
        self.embed_dim = p['embed_dim']
        self.layers = p['layers']
        self.head_dim = p['head_dim']
        self.dropout = p['dropout']
        self.sto_depth = p['sto_depth']

        self.patch_embed = nn.Conv3d(self.in_c, self.embed_dim, (8, 8, 8), (8, 8, 8), 0, bias=False)
        self.patch_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False)
        self.position_embed = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.layers = nn.ModuleList([
            TransformerLayer(self.embed_dim, self.head_dim, self.layers, bias=True,
                             dropout=self.dropout, sto_depth=self.sto_depth)
            for _ in range(self.layers)
        ])
        self.out_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False)
        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(self.embed_dim, self.out_c, (2, 2, 4), (2, 2, 4), 0, bias=False),
            nn.Upsample(scale_factor=(4, 4, 2), mode='trilinear', align_corners=False))
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        B, C, S1, S2, S3 = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, S1*S2*S3, C)
        x = self.patch_norm(x)

        # Positional embeddings - cleanly in and out of residual stream
        pos_embed = get_1d_sinusoidal_pos_embed(S1*S2*S3, self.embed_dim, x.device)
        pos_embed = self.position_embed(pos_embed)
        x = x + pos_embed   # Add positional embedding

        for layer in self.layers:
            x = layer(x)

        x = x - pos_embed   # Remove positional embedding

        x = self.out_norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, S1, S2, S3)
        x = self.out_conv(x)
        return x
    

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    
    B, S1, S2, S3 = 1, 192, 192, 96
    params = {
        "in_channels": 1,
        "out_channels": 14,
        "embed_dim": 512,
        "layers": 8,
        "head_dim": 64,
        "dropout": 0.2,
        "sto_depth": 0.1
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    model = ViTSeg(params).to(device)

    # Profile the forward and backward pass
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        with torch.inference_mode():
            model.eval()
            # with torch.autocast('cuda', torch.bfloat16):
            y = model(x)
        # with torch.autocast('cuda', torch.bfloat16):
        #     y = model(x)
        #     loss = y.sum()
        # loss.backward()

    # assert y.shape == (B, params["out_channels"], S1, S2, S3), "Output shape mismatch"
        
    print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=12))
    if device == torch.device("cuda"):
        print(f"Max VRAM usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", round(total_params / 1e6, 2), 'M')
    
    # Calculate I/O sizes for input and output
    input_size_mb = x.element_size() * x.nelement() / 1024 / 1024
    output_size_mb = y.element_size() * y.nelement() / 1024 / 1024
    print("Input is size:", input_size_mb, 'MB')
    print("Output is size:", output_size_mb, 'MB')