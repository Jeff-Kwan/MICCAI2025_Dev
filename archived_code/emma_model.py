from torch.optim.swa_utils import AveragedModel ,get_ema_multi_avg_fn
model = AveragedModel(model, get_ema_multi_avg_fn(0.9), use_buffers=True)

