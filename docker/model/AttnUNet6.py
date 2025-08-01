import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int, 
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.conv1 =  nn.Sequential(
            nn.Conv3d(in_c, h_c, 1, 1, 0, bias=bias),
            nn.Conv3d(h_c, h_c, 3, 1, 1, bias=bias, groups=h_c))
        self.conv2 = nn.Sequential(
            nn.Conv3d(h_c, h_c, 1, 1, 0, bias=bias),
            nn.Conv3d(h_c, h_c, 3, 1, 1, bias=bias, groups=h_c))
        self.out_conv = nn.Sequential(
            nn.GroupNorm(h_c*2, h_c*2),
            nn.GELU(),
            nn.Conv3d(h_c*2, out_c, 1, 1, 0, bias=False))
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x, self.conv2(x)], dim=1)
        return self.out_conv(x)


class ConvLayer(nn.Module):
    def __init__(self, in_c: int, conv: int, repeats: int, bias: bool = True, 
                 dropout: float = 0.0):
        super().__init__()
        self.repeats = repeats
        self.convs = nn.ModuleList([
            ConvBlock(in_c, conv, in_c, bias, dropout)
            for _ in range(repeats)])

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x
    
class SwiGLU(nn.Module):
    def __init__(self, in_c: int, h_c: int, out_c: int,
                 bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(in_c, h_c * 2, bias)
        self.act = nn.SiLU()
        self.linear2 = nn.Sequential(
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(h_c, out_c, bias=False))
        
    def forward(self, x):
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        x = self.linear2(self.act(x1) * x2)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, in_c: int, head_dim: int, repeats: int, bias: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        assert in_c % head_dim == 0, "in_c must be divisible by head_dim"
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
        B, C, S1, S2, S3 = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, S1*S2*S3, C)
        for norm, mha, mlp in zip(self.mha_norms, self.MHAs, self.mlps):
            norm_x = norm(x)
            x = x + mha(norm_x, norm_x, norm_x, need_weights=False)[0]
            x = x + mlp(x)
        x = x.permute(0, 2, 1).reshape(B, C, S1, S2, S3)
        return x


class Encoder(nn.Module):
    def __init__(self, channels: list, convs: list, layers: list, dropout: float = 0.0):
        super().__init__()
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"
        self.stages = len(channels)
        self.encoder_convs = nn.ModuleList(
            [nn.Sequential(
                ConvLayer(channels[i], convs[i], layers[i], bias=False, 
                          dropout=dropout * (i+1) / self.stages),
                nn.GroupNorm(channels[i]//8, channels[i], affine=False))
             for i in range(self.stages - 1)])
        self.downs = nn.ModuleList([nn.Conv3d(channels[i], channels[i+1], 2, 2, 0, bias=False)
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
        self.stages = len(channels)
        self.decoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, 
                       dropout=dropout * (i+1) / self.stages)
             for i in reversed(range(self.stages - 1))])
        self.ups = nn.ModuleList([nn.Sequential(
                nn.GroupNorm(channels[i+1]//8, channels[i+1], affine=False),
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


class AttnUNet6(nn.Module):
    def __init__(self, p: dict):
        super().__init__()
        self.model_params = p
        channels = p["channels"]
        convs = p["convs"]
        layers = p["e_layers"]
        d_layers = p["d_layers"]
        head_dim = p["head_dim"]
        out_c = p["out_channels"]
        dropout = p.get("dropout", 0.0)
        assert (len(channels) == len(convs) == len(layers)), "Channels, convs, and layers must have the same length"

        self.in_conv = nn.Conv3d(1, channels[0], (2, 2, 1), (2, 2, 1), 0, bias=False)
        
        self.encoder = Encoder(channels, convs, layers, dropout)
        self.bottleneck = nn.Sequential(
            *[nn.Sequential(
                ConvLayer(channels[-1], convs[-1], 1, 
                      bias=False, dropout=dropout),
                TransformerLayer(channels[-1], head_dim, 1,
                        bias=False, dropout=dropout))
                for _ in range(layers[-1])])
        self.decoder = Decoder(channels, convs, d_layers, dropout)

        self.out_conv = nn.Sequential(
            nn.ConvTranspose3d(channels[0], 16, (2, 2, 1), (2, 2, 1), 0, bias=False),
            nn.GroupNorm(1, 16, affine=False),
            nn.Conv3d(16, out_c, (3, 3, 1), 1, (1, 1, 0), bias=True))

        
    def forward(self, x):
        x = self.in_conv(x)

        # Encoder
        x, skips = self.encoder(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        x = self.decoder(x, skips)

        x = self.out_conv(x)
        return x