from monai.losses import DiceLoss, FocalLoss
import torch
from torch.nn import functional as F

B, C, H, W, D = 1, 14, 32, 32, 32
x = torch.randn(B, C, H, W, D)
y = F.softmax(torch.randn(B, C, H, W, D), dim=1)    # Use soft labels

class SoftDiceFocalLoss(DiceLoss, FocalLoss):
    def __init__(self, include_background=True, to_onehot_y=False, softmax=True, weight=None, lambda_focal=1.0, lambda_dice=1.0):
        DiceLoss.__init__(self, include_background=include_background, to_onehot_y=to_onehot_y, softmax=softmax, weight=weight, soft_label=True)
        FocalLoss.__init__(self, include_background=include_background, to_onehot_y=to_onehot_y, use_softmax=softmax, weight=weight)
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice

    def forward(self, inputs, targets):
        dice_loss = super().forward(inputs, targets)
        focal_loss = super().forward(inputs, targets)
        return self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

criterion = SoftDiceFocalLoss(
    include_background=True, 
    to_onehot_y=False, 
    softmax=True, 
    weight=torch.tensor([0.01] + [1.0] * 13))

loss = criterion(x, y)
print(f"Loss: {loss.item()}")