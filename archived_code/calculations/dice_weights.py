import numpy as np

dice = np.array([
0.9723077, 0.9078914, 0.9703501, 0.90021104, 0.9555506, 0.9060197,
0.81629777, 0.7979219, 0.72656524, 0.8356327, 0.9093802, 0.8346007,
0.8948825
])
dice  = 1 - dice    # Weight by relative dice error
weights = dice / np.min(dice)
print(np.round(weights, 3).tolist())