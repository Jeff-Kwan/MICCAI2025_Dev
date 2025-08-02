import torch

x = torch.randn(3, 224, 224, 112)

y = torch.zeros((14, *x.shape[1:]), dtype=torch.uint8)
print(y.shape)