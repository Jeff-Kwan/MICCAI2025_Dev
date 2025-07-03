from monai.losses import DiceLoss, FocalLoss
import torch
import torch.nn as nn
from torch.nn import functional as F

B, C, H, W, D = 1, 14, 16, 16, 16
x = torch.randn(B, C, H, W, D)
y = F.softmax(torch.randn(B, C, H, W, D), dim=1)    # Use soft labels
y = y.to(torch.float16)

class SoftDiceFocalLoss(nn.Module):
    def __init__(self, include_background=True, softmax=True, weight=None, lambda_focal=1.0, lambda_dice=1.0):
        super().__init__()
        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=False,
            softmax=softmax,
            weight=weight,
            soft_label=True)    # Use soft labels
        self.focal_loss = FocalLoss(
            include_background=include_background,
            to_onehot_y=False,
            use_softmax=softmax,
            weight=weight)
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_dice = self.dice_loss(inputs, targets)
        l_focal = self.focal_loss(inputs, targets)
        return self.lambda_dice * l_dice + self.lambda_focal * l_focal

criterion = SoftDiceFocalLoss(
    include_background=True, 
    softmax=True, 
    weight=torch.tensor([0.01] + [1.0] * 13))

loss = criterion(x, y)
print(f"Loss: {loss.item()}")