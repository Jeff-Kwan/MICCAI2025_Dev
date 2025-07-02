import numpy as np
import matplotlib.pyplot as plt

dice_scores1 = np.array([
    0.9723077,
    0.9078914,
    0.9703501,
    0.90021104,
    0.9555506,
    0.9060197,
    0.81629777,
    0.7979219,
    0.72656524,
    0.8356327,
    0.9093802,
    0.8346007,
    0.8948825
])

dice_scores2 = np.array([
    0.974501,
    0.90860415,
    0.9701618,
    0.8948348,
    0.95353776,
    0.90829694,
    0.84161204,
    0.8185945,
    0.8420479,
    0.8282315,
    0.912257,
    0.8343015,
    0.90371466
])

dice_scores3 = np.array([
    0.96879244, 0.9008999, 0.9586724, 0.86094606, 0.9403242, 0.88299036,
    0.76414, 0.73810077, 0.6842463, 0.78051645, 0.90128654, 0.78250253,
    0.89690316
])

plt.figure(figsize=(10, 5))
plt.plot(dice_scores1, marker='o', label='AttnUNet Dice')
plt.plot(dice_scores2, marker='s', label='ConvSeg Dice')
plt.plot(dice_scores3, marker='^', label='ViTSeg Dice')
plt.title('Dice Scores')
plt.xlabel('Sample Index')
plt.ylabel('Dice Score')
plt.ylim(0.65, 1)
plt.grid(True)
plt.legend()
plt.show()


# dice_scores1 = np.array([
#     0.9723077,
#     0.9078914,
#     0.9703501,
#     0.90021104,
#     0.9555506,
#     0.9060197,
#     0.81629777,
#     0.7979219,
#     0.72656524,
#     0.8356327,
#     0.9093802,
#     0.8346007,
#     0.8948825
# ])

# dice_scores2 = np.array([
#     0.974501,
#     0.90860415,
#     0.9701618,
#     0.8948348,
#     0.95353776,
#     0.90829694,
#     0.84161204,
#     0.8185945,
#     0.8420479,
#     0.8282315,
#     0.912257,
#     0.8343015,
#     0.90371466
# ])

# plt.figure(figsize=(10, 5))
# plt.plot(dice_scores1, marker='o', label='Pretrain Dice')
# plt.plot(dice_scores2, marker='s', label='Finetune Dice')
# plt.title('Dice Scores')
# plt.xlabel('Sample Index')
# plt.ylabel('Dice Score')
# plt.ylim(0.65, 1)
# plt.grid(True)
# plt.legend()