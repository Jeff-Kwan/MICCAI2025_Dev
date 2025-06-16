import torch
from monai.losses import DiceFocalLoss, FocalLoss, DiceLoss
from monai.metrics import DiceMetric
import torch.profiler

# criterion = DiceFocalLoss(
#     include_background=True, 
#     to_onehot_y=True, 
#     softmax=True)

# criterion = FocalLoss(include_background=True, to_onehot_y=True)
# criterion = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)

criterion = DiceMetric()

B, C, H, W, D = 1, 14, 256, 256, 256
x = torch.randn(B, C, H, W, D).cuda()
y = torch.randint(0, C, (B, 1, H, W, D)).cuda()

torch.cuda.empty_cache()  # Clear any cached memory before profiling
torch.cuda.reset_peak_memory_stats()  # Reset peak memory stats

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = criterion(x, y)
    torch.cuda.synchronize()  # Ensure all CUDA ops are finished

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
