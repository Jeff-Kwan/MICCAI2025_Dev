import torch
from torch import nn
from torchvision.ops import stochastic_depth


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int, 
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_c, h_c, 1, 1, 0, bias=bias),
            nn.GroupNorm(h_c, h_c),
            nn.Conv3d(h_c, h_c, 3, 1, 1, bias=bias, groups=h_c),
            nn.SiLU(),
            nn.Conv3d(h_c, h_c, 3, 1, 1, bias=bias, groups=h_c),
            nn.Dropout3d(dropout) if dropout else nn.Identity(),
            nn.Conv3d(h_c, out_c, 1, 1, 0, bias=bias))

    def forward(self, x):
        return self.convs(x)


class ConvLayer(nn.Module):
    def __init__(self, in_c: int, conv: int, repeats: int, bias: bool = True, 
                 dropout: float = 0.0):
        super().__init__()
        self.repeats = repeats
        self.convs = nn.ModuleList([
            ConvBlock(in_c, conv, in_c, bias, dropout)
            for _ in range(repeats)])

    def forward(self, x):
        for i in range(self.repeats):
            x = x + self.convs[i](x)
        return x
    
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
            nn.RMSNorm(in_c) for _ in range(repeats)])
        self.MHAs = nn.ModuleList([
            nn.MultiheadAttention(in_c, in_c//head_dim, dropout=dropout, 
                        batch_first=True, bias=bias)
            for _ in range(repeats)])
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.RMSNorm(in_c),
                SwiGLU(in_c, in_c*2, in_c, bias=bias, dropout=dropout))
            for _ in range(repeats)])

    def forward(self, x):
        B, C, S1, S2, S3 = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, S1*S2*S3, C)
        for i in range(self.repeats):
            norm_x = self.mha_norms[i](x)
            x = x + stochastic_depth(self.MHAs[i](norm_x, norm_x, norm_x, need_weights=False)[0], 
                                     self.sto_depth, 'row', self.training)
            x = x + stochastic_depth(self.mlps[i](x), self.sto_depth, 'row', self.training)
        x = x.permute(0, 2, 1).reshape(B, C, S1, S2, S3)
        return x


class Encoder(nn.Module):
    def __init__(self, channels: list, convs: list, layers: list, dropout: float = 0.0):
        super().__init__()
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"
        self.stages = len(channels)
        self.encoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, dropout=dropout)
             for i in range(self.stages - 1)])
        self.downs = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(channels[i], channels[i], affine=False),
                nn.Conv3d(channels[i], channels[i+1], 2, 2, 0, bias=False))
             for i in range(self.stages - 1)])
        
    def forward(self, x):
        skips = []
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x)
            skips.append(x)
            x = self.downs[i](x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, channels: list, convs: list, layers: list, dropout: float = 0.0):
        super().__init__()
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"
        self.stages = len(channels)
        self.decoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, dropout=dropout)
             for i in reversed(range(self.stages - 1))])
        self.ups = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(channels[i+1], channels[i+1], affine=False),
                nn.ConvTranspose3d(channels[i+1], channels[i], 2, 2, 0, bias=False))
             for i in reversed(range(self.stages - 1))])
        self.merges = nn.ModuleList([
             nn.Conv3d(channels[i] * 2, channels[i], 1, 1, 0, bias=False)
             for i in reversed(range(self.stages - 1))])

    def forward(self, x, skips):
        for i, conv in enumerate(self.decoder_convs):
            x = self.ups[i](x)
            x = self.merges[i](torch.cat([x, skips.pop()], dim=1))
            x = conv(x)
        return x


class AttnUNet(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        channels = p["channels"]
        convs = p["convs"]
        layers = p["layers"]
        out_c = p["out_channels"]
        dropout = p.get("dropout", 0.0)
        sto_depth = p.get("stochastic_depth", 0.0)
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"

        self.in_conv = nn.Conv3d(1, channels[0], 2, 2, 0, bias=False)
        
        self.encoder = Encoder(channels, convs, layers, dropout)
        self.bottleneck = TransformerLayer(channels[-1], convs[-1], layers[-1],
                        bias=False, dropout=dropout, sto_depth=sto_depth)
        self.decoder = Decoder(channels, convs, layers, dropout)

        self.out_norm = nn.GroupNorm(channels[0], channels[0], affine=False)
        self.out_conv = nn.ConvTranspose3d(channels[0], out_c, 2, 2, 0, bias=False)

        
    def forward(self, x):
        x = self.in_conv(x)

        # Encoder
        x, skips = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x, skips)

        x = self.out_norm(x)
        x = self.out_conv(x)
        return x

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cpu")
    
    B, S1, S2, S3 = 1, 224, 224, 112
    params = {
        "out_channels": 14,
        "channels":     [32, 64, 128, 256],
        "convs":        [16, 32, 64, 32],
        "layers":       [2, 2, 2, 8],
        "dropout":      0.05,
        "stochastic_depth": 0.1
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    model = AttnUNet(params).to(device)

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