import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import checkpoint
from torchvision.ops import stochastic_depth

# ---------- normalization ----------------------------------------------------

class LayerNormTranspose(nn.Module):
    """
    Wrapper that lets LayerNorm act over an arbitrary dimension by
    temporarily transposing it to the last position.
    """
    def __init__(self, dim: int, features: int, eps: float = 1e-6,
                 elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(features, eps, elementwise_affine)

    def forward(self, x):
        # (..., C, ...) -> (..., ..., C) -> norm -> restore
        x = x.transpose(self.dim, -1)
        x = self.norm(x)
        return x.transpose(self.dim, -1)

# ---------- attention ---------------------------------------------------------

class HyperEdgeAttention(nn.Module):
    """Hyper-edge partitioning + MHA   (3-D version)."""
    def __init__(self, channels: int, edges: int, heads: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        if channels % heads:
            raise ValueError(f"{channels=} not divisible by {heads=}")
        
        self.edges = edges
        self.in_norm = nn.LayerNorm(channels)
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


# ---------- convolutional blocks ---------------------------------------------

class ConvBlock(nn.Module):
    """Local + dilated block."""
    def __init__(self, in_c: int, h_c: int, out_c: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.Sequential(
            LayerNormTranspose(1, in_c),
            nn.Conv3d(in_c, h_c, 3, 1, 1, bias=bias),
            nn.SiLU(),
            nn.Dropout3d(dropout) if dropout else nn.Identity(),
            nn.Conv3d(h_c, out_c, 3, 1, 1, bias=bias))
        
    def _inner(self, x):
        return self.convs(x)

    def forward(self, x):
        if self.training and x.requires_grad:
            return checkpoint.checkpoint(self._inner, x, use_reentrant=False)
        else:
            return self._inner(x)

class SwiGLU(nn.Module):
    """Channel-wise SwiGLU MLP."""
    def __init__(self, in_c: int, h_c: int, out_c: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.in_norm = LayerNormTranspose(1, in_c)
        self.conv1 = nn.Conv3d(in_c, h_c * 2, 1, 1, 0, bias=bias)
        self.act = nn.SiLU()
        self.conv2 = nn.Sequential(
            nn.Dropout3d(dropout) if dropout else nn.Identity(),
            nn.Conv3d(h_c, out_c, 1, 1, 0, bias=bias))
        
    def _inner(self, x):
        x1, x2 = self.conv1(x).chunk(2, dim=1)
        return self.conv2(self.act(x1) * x2)

    def forward(self, x):
        x = self.in_norm(x)
        if self.training and x.requires_grad:
            return checkpoint.checkpoint(self._inner, x, use_reentrant=False)
        else:
            return self._inner(x)


# ---------- composite transformer-like layer ---------------------------------

class Layer(nn.Module):
    def __init__(self, in_c: int, conv: int, attn: list | int, mlp: int,
                 bias: bool = False, dropout: float = 0.0, sto_depth: float = 0.0):
        super().__init__()
        self.sto_depth = sto_depth
        edge_n, heads = attn if isinstance(attn, (list, tuple)) else (attn, 1)
        self.conv = ConvBlock(in_c, conv, in_c, bias, dropout)
        self.attn = HyperEdgeAttention(in_c, edge_n, heads, bias, dropout)
        self.mlp  = SwiGLU(in_c, mlp, in_c, bias, dropout)

    def forward(self, x):
        x = x + stochastic_depth(self.conv(x), self.sto_depth, 'row', self.training)
        x = x + stochastic_depth(self.attn(x), self.sto_depth, 'row', self.training)
        x = x + stochastic_depth(self.mlp(x), self.sto_depth, 'row', self.training)
        return x


# ---------- full model --------------------------------------------------------

class HarmonicEncoder(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        in_c = p["channels"][0]
        channels = p["channels"][1:]
        convs = p["convs"][1:]
        attns = p["attns"]
        mlps = p["mlps"]
        layers = p["layers"]
        dropout = p.get("dropout", 0.0)
        sto_depth = p.get("stochastic_depth", 0.0)

        # Helper to build downsample + encoder blocks
        def make_stage(in_ch, out_ch, conv, attn, mlp, n_layers):
            down = nn.Sequential(
                LayerNormTranspose(1, in_ch, elementwise_affine=False, bias=False),
                nn.Conv3d(in_ch, out_ch, 2, 2, 0, bias=False)
            )
            enc = nn.ModuleList([
                Layer(out_ch, conv, attn, mlp, dropout=dropout, sto_depth=sto_depth)
                for _ in range(n_layers)
            ])
            return down, enc

        self.down1, self.encoder1 = make_stage(in_c, channels[0], convs[0], attns[0], mlps[0], layers[0])
        self.down2, self.encoder2 = make_stage(channels[0], channels[1], convs[1], attns[1], mlps[1], layers[1])
        self.down3, self.encoder3 = make_stage(channels[1], channels[2], convs[2], attns[2], mlps[2], layers[2])

    def forward(self, x, skips=False):
        skip_list = [x] if skips else None

        x = self.down1(x)
        for layer in self.encoder1:
            x = layer(x)
        if skips: skip_list.append(x)

        x = self.down2(x)
        for layer in self.encoder2:
            x = layer(x)
        if skips: skip_list.append(x)

        x = self.down3(x)
        for layer in self.encoder3:
            x = layer(x)

        return (x, skip_list) if skips else (x, None)


class HarmonicDecoder(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        in_c = p["channels"][0]
        channels = p["channels"][1:]
        convs = p["convs"][1:]
        attns = p["attns"]
        mlps = p["mlps"]
        layers = p["layers"]
        dropout = p.get("dropout", 0.0)
        sto_depth = p.get("stochastic_depth", 0.0)

        # Helper to build decoder stage
        def make_decoder_stage(in_ch, out_ch, conv, attn, mlp, n_layers):
            dec = nn.ModuleList([
                Layer(in_ch, conv, attn, mlp, dropout=dropout, sto_depth=sto_depth)
                for _ in range(n_layers)
            ])
            up = nn.Sequential(
                LayerNormTranspose(1, in_ch, elementwise_affine=False, bias=False),
                nn.ConvTranspose3d(in_ch, out_ch, 2, 2, 0, bias=False)
            )
            merge = nn.Conv3d(out_ch, out_ch, 1, 1, 0, bias=False)
            return dec, up, merge

        self.decoder3, self.up3, self.merge3 = make_decoder_stage(
            channels[2], channels[1], convs[2], attns[2], mlps[2], layers[2])
        self.decoder2, self.up2, self.merge2 = make_decoder_stage(
            channels[1], channels[0], convs[1], attns[1], mlps[1], layers[1])
        self.decoder1, self.up1, self.merge1 = make_decoder_stage(
            channels[0], in_c, convs[0], attns[0], mlps[0], layers[0])

    def forward(self, x, skips=False):
        for layer in self.decoder3:
            x = layer(x)
        x = self.up3(x)
        if skips: x = x + self.merge3(skips.pop())

        for layer in self.decoder2:
            x = layer(x)
        x = self.up2(x)
        if skips: x = x + self.merge2(skips.pop())

        for layer in self.decoder1:
            x = layer(x)
        x = self.up1(x)
        if skips: x = x + self.merge1(skips.pop())

        return x


class VAEPrior(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.in_conv = nn.Conv3d(1, p['channels'][0], 1, 1, 0, bias=False)
        self.init_convs = nn.ModuleList(
            [ConvBlock(p['channels'][0], p['convs'][0], p['channels'][0], dropout=p['dropout']) 
             for _ in range(2)])
        self.final_convs = nn.ModuleList(
            [ConvBlock(p['channels'][0], p['convs'][0], p['channels'][0], dropout=p['dropout'])
                for _ in range(2)])
        self.out_conv = nn.Sequential(
            LayerNormTranspose(1, p["channels"][0], elementwise_affine=False, bias=False),
            nn.Conv3d(p["channels"][0], p["out_channels"], 1, 1, 0, bias=False))
        
        self.encoder = HarmonicEncoder(p)
        self.mu_var = nn.Sequential(
            LayerNormTranspose(1, p["channels"][-1], elementwise_affine=False, bias=False),
            nn.Conv3d(p["channels"][-1], p["channels"][1] * 2, 1, 1, 0, bias=False))
        self.decoder = HarmonicDecoder(p)
        
        
    def encode(self, x):
        x, _ = self.encoder(x, skips=False)
        mu, var = self.mu_var(x).chunk(2, dim=1)
        return mu, var
    
    def reparameterize(self, mu, var):
        eps = torch.randn_like(mu, device=mu.device)
        return mu + eps * torch.exp(0.5 * var)
    
    def decode(self, z):
        return self.decoder(z, skips=False)
    
    def forward(self, x):
        x = self.in_conv(x)
        
        # Initial convolutions
        for conv in self.init_convs:
            x = x + conv(x)

        # Encoder
        mu, var = self.encode(x)
        z = self.reparameterize(mu, var)

        # Decoder
        x = self.decode(z)

        # Final output convolution
        x = self.out_conv(x)
        return x, mu, var



class HarmonicSeg(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        in_c = p["channels"][0]
        out_c = p["out_channels"]
        dropout = p.get("dropout", 0.0)
        
        # Native Resolution
        self.in_conv = nn.Conv3d(1, in_c, 1, 1, 0, bias=False)
        self.init_convs = nn.ModuleList(
            [ConvBlock(in_c, p['convs'][0], in_c, dropout=dropout) for _ in range(2)])
        
        self.final_convs = nn.ModuleList(
            [ConvBlock(in_c, p['convs'][0], in_c, dropout=dropout) for _ in range(2)])
        self.out_conv = nn.Sequential(
            LayerNormTranspose(1, in_c, elementwise_affine=False, bias=False),
            nn.ConvTranspose3d(in_c, out_c, 1, 1, 0, bias=False))
        
        # Encoder & Decoder
        self.encoder = HarmonicEncoder(p)
        self.decoder = HarmonicDecoder(p)
        
    def forward(self, x):
        x = self.in_conv(x)
        
        # Initial convolutions
        for conv in self.init_convs:
            x = x + conv(x)

        # Encoder
        x, skips = self.encoder(x, skips=True)

        # Decoder
        x = self.decoder(x, skips=skips)

        # Final Refining
        for conv in self.final_convs:
            x = x + conv(x)
        x = self.out_conv(x)
        return x

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    
    B, S1, S2, S3 = 1, 160, 160, 80
    params = {
        "out_channels": 14,
        "channels":     [16, 64, 128, 256],
        "convs":        [32, 64, 64, 64],
        "attns":        [[256, 4], [256, 4], [256, 4]],
        "mlps":         [128, 256, 512],
        "layers":       [2, 2, 2],
        "dropout":      0.1,
        "stochastic_depth": 0.1,
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    model = HarmonicSeg(params).to(device)

    # Profile the forward and backward pass
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True
    ) as prof:
        # with torch.inference_mode():
        #     model.eval()
        #     y = model(x)
        with torch.autocast('cuda', torch.bfloat16):
            y = model(x)
            loss = y.sum()
        loss.backward()

    assert y.shape == (B, params["out_channels"], S1, S2, S3), "Output shape mismatch"
        
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