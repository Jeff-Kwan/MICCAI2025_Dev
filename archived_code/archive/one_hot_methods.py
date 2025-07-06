import numpy as np
import torch
from torch.nn.functional import one_hot
from monai.transforms import AsDiscrete
import time

data_pred = torch.randn(14, 128, 128, 128)

trials = 10

# Method 1: Manual numpy one-hot
start = time.time()
for _ in range(trials):
    pred = (np.eye(14, dtype=(np.float32)))[data_pred.argmax(dim=0).numpy()]
    pred = np.moveaxis(pred, -1, 0)
manual_time = (time.time() - start) / trials

# Method 2: torch one_hot
start = time.time()
for _ in range(trials):
    oh_data_pred = one_hot(data_pred.argmax(dim=0), num_classes=14).permute(3, 0, 1, 2)
torch_time = (time.time() - start) / trials

# Method 3: AsDiscrete
# data_pred = data_pred.numpy()
start = time.time()
for _ in range(trials):
    monai_pred = AsDiscrete(argmax=True, to_onehot=14)(data_pred)
monai_time = (time.time() - start) / trials

assert np.all(pred == oh_data_pred.numpy()), "Mismatch between manual and torch one-hot encoding"
assert np.all(pred == monai_pred.numpy()), "Mismatch between manual and MONAI AsDiscrete encoding"

print(f"Manual numpy one-hot time: {manual_time:.6f} seconds")
print(f"Torch one_hot time: {torch_time:.6f} seconds")
print(f"MONAI AsDiscrete time: {monai_time:.6f} seconds")