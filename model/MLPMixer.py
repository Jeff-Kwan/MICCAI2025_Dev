import torch
from torch import nn
from torchvision.ops import stochastic_depth

class ChannelMLP(nn.Module):
    def __init__(self, input_dim, expand, bias=True, dropout=0.0):
        super(ChannelMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim*expand, bias=bias),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(input_dim*expand, input_dim, bias=bias))

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class SpatialMLP(nn.Module):
    def __init__(self, dims, bias=True, dropout=0.0):
        super(SpatialMLP, self).__init__()
        self.norm = nn.LayerNorm(dims[0])
        self.mix = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim, bias=bias),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout else nn.Identity())
            for dim in dims[1:]])
        self.merge = nn.Linear(dims[0]*(len(dims)-1), dims[0], bias=bias)

    def forward(self, x):
        norm_x = self.norm(x)
        x_mix = []
        for i, mix in enumerate(self.mix):
            x_mix.append(mix(norm_x.transpose(i+1, -1)).transpose(i+1, -1))
        x = self.merge(torch.cat(x_mix, dim=-1))
        return x
    

class MLPLayer(nn.Module):
    def __init__(self, dims, layers, bias=True, dropout=0.0, sto_depth=0.0):
        super(MLPLayer, self).__init__()
        self.sto_depth = sto_depth
        self.layers = layers
        self.spatialmlp = nn.ModuleList([
            SpatialMLP(dims, bias=bias, dropout=dropout)
            for _ in range(layers)])
        self.channelmlp = nn.ModuleList([
            ChannelMLP(dims[0], 4, bias=bias, dropout=dropout)
            for _ in range(layers)])
        
    def forward(self, x):
        for spatial_mlp, channel_mlp in zip(self.spatialmlp, self.channelmlp):
            x = x + stochastic_depth(spatial_mlp(x), self.sto_depth, 'row', self.training)
            x = x + stochastic_depth(channel_mlp(x), self.sto_depth, 'row', self.training)
        return x

    
class MLPMixer(nn.Module):
    def __init__(self, p):
        super(MLPMixer, self).__init__()
        self.model_params = p
        dims = p["dims"]
        patch = p["patch"]
        layers = p["layers"]
        out_c = p["out_channels"]
        dropout = p.get("dropout", 0.0)
        sto_depth = p.get("stochastic_depth", 0.0)

        dims = [dims[0]] + [d//p for d, p in zip(dims[1:], patch)]

        self.in_conv = nn.Conv3d(1, dims[0], patch, patch, 0, bias=False)
        self.layers = MLPLayer(dims, layers, bias=True, dropout=dropout, sto_depth=sto_depth)
        self.out_norm = nn.LayerNorm(dims[0], elementwise_affine=False, bias=False)
        self.out_conv = nn.ConvTranspose3d(dims[0], out_c, patch, patch, 0, bias=False)

    def forward(self, x):
        x = self.in_conv(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # Change to (B, S1, S2, S3, C)
        x = self.layers(x)  # Apply MLP layers
        x = self.out_norm(x).permute(0, 4, 1, 2, 3)
        x = self.out_conv(x)
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda")
    
    B, S1, S2, S3 = 1, 224, 224, 112
    params = {
        "out_channels": 14,
        "dims":     [64, S1, S2, S3],
        "patch":    [2, 2, 2],  # Patch size for downsampling
        "layers":       8,
        "dropout":      0.1,
        "stochastic_depth": 0.05
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    model = MLPMixer(params).to(device)

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