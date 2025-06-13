import torch
from torch import nn
from torch.utils import checkpoint
from torchvision.ops import stochastic_depth


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv3d(in_c, h_c, 3, 1, 1, bias=bias),
            nn.GroupNorm(h_c, h_c),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout else nn.Identity(),
            nn.Conv3d(h_c, out_c, 3, 1, 1, bias=bias))
        
    def _inner(self, x):
        return self.convs(x)

    def forward(self, x):
        if self.training and x.requires_grad:
            return checkpoint.checkpoint(self._inner, x, use_reentrant=False)
        else:
            return self._inner(x)


class Layer(nn.Module):
    def __init__(self, in_c: int, conv: int, repeats: int, bias: bool = True, 
                 dropout: float = 0.0, sto_depth: float = 0.0):
        super().__init__()
        self.sto_depth = sto_depth
        self.repeats = repeats
        self.convs = nn.ModuleList([
            nn.ModuleList([
                ConvBlock(in_c, conv, in_c, bias, dropout),
                ConvBlock(in_c, conv, in_c, bias, dropout)])
            for _ in range(repeats)
        ])

    def forward(self, x):
        for i in range(self.repeats):
            x = x + stochastic_depth(self.convs[i][0](x), self.sto_depth, 'row', self.training)
            x = x + stochastic_depth(self.convs[i][1](x), self.sto_depth, 'row', self.training)
        return x


class UNetControl(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        channels = p["channels"]
        convs = p["convs"]
        layers = p["layers"]
        out_c = p["out_channels"]
        dropout = p.get("dropout", 0.0)
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"
        stages = len(channels)

        self.in_conv = nn.Conv3d(1, channels[0], (2, 2, 1), (2, 2, 1), 0, bias=False)

        # Encoder
        self.encoder_convs = nn.ModuleList(
            [Layer(channels[i], convs[i], layers[i], bias=False, dropout=dropout)
             for i in range(stages)])
        self.downs = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(channels[i], channels[i], affine=False),
                nn.Conv3d(channels[i], channels[i+1], 2, 2, 0, bias=False))
             for i in range(stages - 1)])
        
        # Decoder
        self.decoder_convs = nn.ModuleList(
            [Layer(channels[i], convs[i], layers[i], bias=False, dropout=dropout)
             for i in reversed(range(stages))])
        self.ups = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(channels[i+1], channels[i+1], affine=False),
                nn.ConvTranspose3d(channels[i+1], channels[i], 2, 2, 0, bias=False))
             for i in reversed(range(stages - 1))])
        self.merges = nn.ModuleList([
             nn.Conv3d(channels[i] * 2, channels[i], 1, 1, 0, bias=False)
             for i in reversed(range(stages - 1))])

        self.out_conv = nn.Sequential(
            nn.GroupNorm(channels[0], channels[0], affine=False),
            nn.ConvTranspose3d(channels[0], out_c, (2, 2, 1), (2, 2, 1), 0, bias=False))

        
    def forward(self, x):
        x = self.in_conv(x)

        # Encoder
        skips = []
        for i, conv in enumerate(self.encoder_convs):
            x = conv(x)
            if i < len(self.downs):
                skips.append(x)
                x = self.downs[i](x)

        # Decoder
        for i, conv in enumerate(self.decoder_convs):
            x = conv(x)
            if i < len(self.ups):
                x = self.ups[i](x)
                x = self.merges[i](torch.cat([x, skips.pop()], dim=1))

        x = self.out_conv(x)
        return x

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    
    B, S1, S2, S3 = 1, 160, 160, 80
    params = {
        "out_channels": 14,
        "channels":     [32, 64, 128, 256],
        "convs":        [24, 32, 48, 64],
        "layers":       [2, 2, 2, 2],
        "dropout":      0.1,
        "stochastic_depth": 0.1,
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    model = UNetControl(params).to(device)

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