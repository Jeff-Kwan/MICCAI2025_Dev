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
        self.in_conv = nn.Sequential(
            LayerNormTranspose(1, in_c),
            nn.Conv3d(in_c, h_c, 3, 1, 1, bias=bias))
        
        self.dilated = nn.Conv3d(h_c, h_c, 3, 1, 2, bias=bias, dilation=2)

        self.out_conv = nn.Sequential(
            nn.SiLU(),
            nn.Dropout3d(dropout) if dropout else nn.Identity(),
            nn.Conv3d(h_c, out_c, 3, 1, 1, bias=bias))
        
    def _inner(self, x):
        z = self.in_conv(x)
        z = z + self.dilated(z)
        x = self.out_conv(z)
        return x

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


class Downsampling(nn.Module):
    """Downsampling block."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm = LayerNormTranspose(1, in_c, elementwise_affine=False, bias=False)
        self.down = nn.Conv3d(in_c, out_c, 2, 2, 0, bias=False)
        nn.init.kaiming_normal_(self.down.weight, nonlinearity="linear")

    def forward(self, x):
        x = self.norm(x)
        x = self.down(x)
        return x
    

class Upsampling(nn.Module):
    """Upsampling block."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.norm = LayerNormTranspose(1, in_c, elementwise_affine=False, bias=False)
        self.up = nn.ConvTranspose3d(in_c, out_c, 2, 2, 0, bias=False)
        nn.init.kaiming_normal_(self.up.weight, nonlinearity="linear")

    def forward(self, x):
        x = self.norm(x)
        x = self.up(x)
        return x


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


# ---------- Positional embedding ----------------------------------

class PositionEmbedding(nn.Module):
    """
    Applies a fixed sin-cos positional embedding.
    Caches the positional embedding buffer for efficiency.
    """
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4."
        self.channels = channels
        # fixed sin-cos on z, y, x
        self.pos_embed = nn.Conv3d(channels*3//2, channels, 1, 1, 0, bias=False)
        self.norm = LayerNormTranspose(1, channels, elementwise_affine=False, bias=False)
        self.register_buffer("rads", 2048 ** torch.linspace(0, 1, channels // 4).view(-1, 1, 1, 1),
                             persistent=False)
        self.register_buffer("_cached_pos", None, persistent=False)
        self._cached_shape = None

    @torch.no_grad()
    def _build_3d_sincos(self, D: int, H: int, W: int, device):
        z = torch.linspace(0, 1, D, device=device).view(1, D, 1, 1)
        y = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        x = torch.linspace(0, 1, W, device=device).view(1, 1, 1, W)

        z_embed = self.rads * z
        y_embed = self.rads * y
        x_embed = self.rads * x

        pos = torch.cat([
            torch.sin(z_embed).expand(1, -1, D, H, W),
            torch.cos(z_embed).expand(1, -1, D, H, W),
            torch.sin(y_embed).expand(1, -1, D, H, W),
            torch.cos(y_embed).expand(1, -1, D, H, W),
            torch.sin(x_embed).expand(1, -1, D, H, W),
            torch.cos(x_embed).expand(1, -1, D, H, W),
        ], dim=1)  # (1, C*3//2, D, H, W)
        return pos

    def forward(self, x):
        shape = (1, self.channels * 3 // 2, *x.shape[2:])
        if (self._cached_pos is None or self._cached_shape != shape):
            pos = self._build_3d_sincos(*x.shape[2:], x.device)
            self._cached_pos = pos
            self._cached_shape = shape
        else:
            pos = self._cached_pos
        x = self.norm(x + self.pos_embed(pos))
        return x


# ---------- full model --------------------------------------------------------

class HarmonicSeg(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        in_c = p["in_channels"]
        out_c = p["out_channels"]
        channels = p["channels"]
        convs = p["convs"]
        attns = p["attns"]
        mlps = p["mlps"]
        layers = p["layers"]
        dropout = p.get("dropout", 0.0)
        sto_depth = p.get("stochastic_depth", 0.0)
        if not (len(channels) == len(convs) == len(attns) == len(mlps) == len(layers) == 3):
            raise ValueError("All lists must be length 3.")
        
        # Native Resolution
        self.in_conv = nn.Conv3d(1, in_c, 2, 2, 0, bias=False)
        self.init_convs = nn.ModuleList(
            [ConvBlock(in_c, in_c, in_c, dropout=dropout) for _ in range(2)])
        
        self.merge_final = nn.Conv3d(in_c*2, in_c, 1, 1, 0, bias=False)
        self.final_convs = nn.ModuleList(
            [ConvBlock(in_c, in_c, in_c, dropout=dropout) for _ in range(2)])
        self.out_conv = nn.Sequential(
            LayerNormTranspose(1, in_c, elementwise_affine=False, bias=False),
            nn.Conv3d(in_c, out_c, 1, 1, 0, bias=False))

        # Down Once Scale
        self.position_embed = PositionEmbedding(channels[0])
        self.down1 = Downsampling(in_c, channels[0])
        self.encoder1 = nn.ModuleList(
            [Layer(channels[0], convs[0], attns[0], mlps[0], dropout=dropout, sto_depth=sto_depth)
            for _ in range(layers[0])])
        
        self.merge1 = nn.Conv3d(channels[0]*2, channels[0], 1, 1, 0, bias=False)
        self.decoder1 = nn.ModuleList(
            [Layer(channels[0], convs[0], attns[0], mlps[0], dropout=dropout, sto_depth=sto_depth)
            for _ in range(layers[0])])
        self.up1 = Upsampling(channels[0], in_c)

        # Down Twice Scale
        self.down2 = Downsampling(channels[0], channels[1])
        self.encoder2 = nn.ModuleList(
            [Layer(channels[1], convs[1], attns[1], mlps[1], dropout=dropout, sto_depth=sto_depth)
            for _ in range(layers[1])])
        
        self.merge2 = nn.Conv3d(channels[1]*2, channels[1], 1, 1, 0, bias=False)
        self.decoder2 = nn.ModuleList(
            [Layer(channels[1], convs[1], attns[1], mlps[1], dropout=dropout, sto_depth=sto_depth)
            for _ in range(layers[1])])
        self.up2 = Upsampling(channels[1], channels[0])
        
        # Down Thrice Scale
        self.down3 = Downsampling(channels[1], channels[2])
        self.encoder3 = nn.ModuleList(
            [Layer(channels[2], convs[2], attns[2], mlps[2], dropout=dropout, sto_depth=sto_depth)
            for _ in range(layers[2])])
            # Effectively the bottleneck layer
        self.decoder3 = nn.ModuleList(
            [Layer(channels[2], convs[2], attns[2], mlps[2], dropout=dropout, sto_depth=sto_depth)
            for _ in range(layers[2])])
        self.up3 = Upsampling(channels[2], channels[1])

        
    def forward(self, x):
        B, _, S1, S2, S3 = x.shape
        x = self.in_conv(x)

        # Initial convolutions
        for conv in self.init_convs:
            x = x + conv(x)
        skips = [x]  # Store initial feature map for skip connections
        
        # Down Once & Position embedding
        x = self.down1(x)  # Downsample to first channel size
        x = self.position_embed(x)
        for layer in self.encoder1:
            x = layer(x)
        skips.append(x)

        # Down Twice
        x = self.down2(x)
        for layer in self.encoder2:
            x = layer(x)
        skips.append(x)

        # Down Thrice / Up Thrice
        x = self.down3(x)
        for layer in self.encoder3:
            x = layer(x)
        
        for layer in self.decoder3:
            x = layer(x)
        x = self.up3(x)

        # Up Twice
        x = self.merge2(torch.cat([x, skips.pop()], dim=1))
        for layer in self.decoder2:
            x = layer(x)
        x = self.up2(x)

        # Up Once
        x = self.merge1(torch.cat([x, skips.pop()], dim=1))
        for layer in self.decoder1:
            x = layer(x)
        x = self.up1(x)

        # Final Native Refining
        x = self.merge_final(torch.cat([x, skips.pop()], dim=1))
        for conv in self.final_convs:
            x = x + conv(x)
        x = self.out_conv(x)

        x = F.interpolate(x, size=(S1, S2, S3), mode='trilinear', align_corners=False)
        return x

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    
    B, S1, S2, S3 = 1, 128, 128, 128
    params = {
        "in_channels":  16,
        "out_channels": 14,
        "channels":     [64, 128, 256],
        "convs":        [64, 64, 64],
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