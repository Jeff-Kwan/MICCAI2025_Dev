import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils import checkpoint
from torchvision.ops import stochastic_depth

class LayerNormTranspose(nn.Module):
    def __init__(self, dim: int, features: int, eps: float = 1e-6,
                 elementwise_affine: bool = True, bias: bool = True):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(features, eps, elementwise_affine, bias)

    def forward(self, x):
        # (..., C, ...) -> (..., ..., C) -> norm -> restore
        x = x.transpose(self.dim, -1)
        x = self.norm(x)
        return x.transpose(self.dim, -1)

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
        # if self.training and x.requires_grad:
        #     return checkpoint.checkpoint(self._inner, x, use_reentrant=False)
        # else:
        return self._inner(x)


class ConvLayer(nn.Module):
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
                nn.Linear(in_c, in_c*4, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout) if dropout else nn.Identity(),
                nn.Linear(in_c*4, in_c, bias=bias))
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


class VAEPrior(nn.Module):
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
        self.stages = len(channels)
        self.classes = out_c + 1 # + 1 for masking

        # self.img_conv = nn.Conv3d(1, channels[0], 2, 2, 0, bias=False)
        self.label_conv = nn.Conv3d(self.classes, channels[0], 2, 2, 0, bias=False)
        
        # Encoder
        self.encoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, dropout=dropout, sto_depth=sto_depth)
             for i in range(self.stages - 1)])
        self.downs = nn.ModuleList([nn.Sequential(
                LayerNormTranspose(1, channels[i], elementwise_affine=False, bias=False),
                nn.Conv3d(channels[i], channels[i+1], 2, 2, 0, bias=False))
             for i in range(self.stages - 1)])
        self.bottleneck1 = TransformerLayer(channels[-1], convs[-1], layers[-1],
                bias=True, dropout=dropout, sto_depth=sto_depth)
        self.muvar_norm = nn.LayerNorm(channels[-1], elementwise_affine=False, bias=False)
        self.mu_var = nn.Conv3d(channels[-1], channels[-1] * 2, 1, 1, 0, bias=False)

        # Decoder
        self.bottleneck2 = TransformerLayer(channels[-1], convs[-1], layers[-1],
                bias=True, dropout=dropout, sto_depth=sto_depth)
        self.decoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, dropout=dropout, sto_depth=sto_depth)
             for i in reversed(range(self.stages - 1))])
        self.ups = nn.ModuleList([nn.Sequential(
                LayerNormTranspose(1, channels[i+1], elementwise_affine=False, bias=False),
                nn.ConvTranspose3d(channels[i+1], channels[i], 2, 2, 0, bias=False))
             for i in reversed(range(self.stages - 1))])
        self.out_norm = LayerNormTranspose(1, channels[0], elementwise_affine=False, bias=False)
        self.out_conv = nn.ConvTranspose3d(channels[0], out_c, 2, 2, 0, bias=False)
        

    def encode(self, label):
        label = F.one_hot(label, self.classes).squeeze(1).float().permute(0, 4, 1, 2, 3)
        x = self.label_conv(label)

        for i, conv in enumerate(self.encoder_convs):
            x = conv(x)
            x = self.downs[i](x)

        x = self.bottleneck1(x)

        x = self.muvar_norm(x.transpose(1, -1)).transpose(1, -1)
        mu, log_var = self.mu_var(x).chunk(2, dim=1)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        # Reparameterization trick
        return mu + torch.randn_like(mu, device=mu.device) * torch.exp(0.5 * log_var)
    
    def decode(self, x):
        x = self.bottleneck2(x)
        latent_priors = [x]
        for i, conv in enumerate(self.decoder_convs):
            x = self.ups[i](x)
            x = conv(x)
            latent_priors.append(x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        return x, latent_priors
    
    def forward(self, labels):
        mu, log_var = self.encode(labels)
        x = self.reparameterize(mu, log_var)
        x, _ = self.decode(x)
        return x, mu, log_var
    


class VAEPosterior(nn.Module):
    '''
    VAE Posterior based on UNet Control.
    '''
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
        self.stages = len(channels)

        self.in_conv = nn.Conv3d(1, channels[0], 2, 2, 0, bias=False)
        self.vae_prior = VAEPrior(p)
        
        # Encoder
        self.encoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, dropout=dropout, sto_depth=sto_depth)
             for i in range(self.stages - 1)])
        self.downs = nn.ModuleList([nn.Sequential(
                LayerNormTranspose(1, channels[i], elementwise_affine=False, bias=False),
                nn.Conv3d(channels[i], channels[i+1], 2, 2, 0, bias=False))
             for i in range(self.stages - 1)])
        self.bottleneck1 = TransformerLayer(channels[-1], convs[-1], layers[-1],
                bias=True, dropout=dropout, sto_depth=sto_depth)
        self.muvar_norm = nn.LayerNorm(channels[-1], elementwise_affine=False, bias=False)
        self.mu_var = nn.Conv3d(channels[-1], channels[-1] * 2, 1, 1, 0, bias=False)

        # Decoder
        self.bottleneck2 = TransformerLayer(channels[-1], convs[-1], layers[-1],
                bias=True, dropout=dropout, sto_depth=sto_depth)
        self.decoder_convs = nn.ModuleList(
            [ConvLayer(channels[i], convs[i], layers[i], bias=False, dropout=dropout, sto_depth=sto_depth)
             for i in reversed(range(self.stages - 1))])
        self.ups = nn.ModuleList([nn.Sequential(
                LayerNormTranspose(1, channels[i+1], elementwise_affine=False, bias=False),
                nn.ConvTranspose3d(channels[i+1], channels[i], 2, 2, 0, bias=False))
             for i in reversed(range(self.stages - 1))])
        self.merge_lat = nn.Conv3d(channels[-1] * 2, channels[-1], 1, 1, 0, bias=False)
        self.merges = nn.ModuleList([
             nn.Conv3d(channels[i] * 3, channels[i], 1, 1, 0, bias=False)
             for i in reversed(range(self.stages - 1))])
        self.out_norm = LayerNormTranspose(1, channels[0], elementwise_affine=False, bias=False)
        self.out_conv = nn.ConvTranspose3d(channels[0], out_c, 2, 2, 0, bias=False)

        
    def img_encode(self, x):
        x = self.in_conv(x)

        skips = []
        for down, conv in zip(self.downs, self.encoder_convs):
            x = conv(x)
            skips.append(x)
            x = down(x)

        x = self.bottleneck1(x)
        mu, log_var = self.mu_var(self.muvar_norm(x.transpose(1, -1)).transpose(1, -1)).chunk(2, dim=1)
        return mu, log_var, skips
    
    def decode(self, x, skips, latent_priors):
        x = self.merge_lat(torch.cat([x, latent_priors.pop(0)], dim=1))
        x = self.bottleneck2(x)

        for up, conv, merge in zip(self.ups, self.decoder_convs, self.merges):
            x = up(x)
            x = merge(torch.cat([x, skips.pop(), latent_priors.pop(0)], dim=1))
            x = conv(x)

        x = self.out_norm(x)
        x = self.out_conv(x)
        return x
    
    def forward(self, img, labels=None):
        if self.training:
            # During training, teacher forcing on vae prior decoding
            mu, log_var = self.vae_prior.encode(labels)
            prior_z = self.vae_prior.reparameterize(mu, log_var)
            prior_x, latent_priors = self.vae_prior.decode(prior_z)

            mu_hat, log_var_hat, skips = self.img_encode(img)
            # latent_priors = [lp.detach().clone().requires_grad_() for lp in latent_priors]
            # skips = [s.detach().clone().requires_grad_() for s in skips]
            # x = self.decode(prior_z.detach().clone().requires_grad_(), skips, latent_priors)
            x = self.decode(prior_z, skips, latent_priors)
            return x, mu_hat, log_var_hat, prior_x, mu, log_var
        else:
            # During inference, latent estimation from image
            _, _, S1, S2, S3 = img.shape
            mu_hat, log_var_hat, skips = self.img_encode(img)
            z_hat = self.vae_prior.reparameterize(mu_hat, log_var_hat)
            x, latent_priors = self.vae_prior.decode(z_hat)
            x = self.decode(z_hat, skips, latent_priors)
            x = F.interpolate(x, size=(S1, S2, S3), mode='trilinear')
            return x

# ---------- demo ----------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda")
    # print("DO NOT RUN ON LAPTOP!")
    # exit()
    
    B, S1, S2, S3 = 1, 416, 224, 128
    params = {
        "out_channels": 14,
        "channels":     [32, 64, 128, 256],
        "convs":        [32, 48, 64, 32],
        "layers":       [2, 2, 2, 2],
        "dropout":      0.1,
        "stochastic_depth": 0.1
    }

    x = torch.randn(B, 1, S1, S2, S3).to(device)
    labels = torch.randint(0, params["out_channels"], (B, 1, S1, S2, S3)).long().to(device)
    model = VAEPosterior(params).to(device)

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
            y = model(x, labels)[0]
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