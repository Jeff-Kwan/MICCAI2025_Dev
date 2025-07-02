import numpy as np

attnunet_dice = np.array([0.9733564, 0.90776205, 0.9711724, 0.90172356, 0.95720917, 0.9067709,
                          0.81953615, 0.7954017, 0.7448793, 0.83528334, 0.91554433, 0.8372468,
                          0.89655006])
convseg_dice = np.array([0.9760091, 0.9082546, 0.97139025, 0.8963372, 0.955216, 0.9093051,
                         0.83761793, 0.81330675, 0.8221831, 0.82891464, 0.91047496, 0.83300036,
                         0.89641935])
vitseg_dice = np.array([0.9682947, 0.9035306, 0.959599, 0.8615064, 0.9409744, 0.8836515,
                        0.7659102, 0.73720634, 0.67131287, 0.7886859, 0.9026979, 0.7868216,
                        0.9012726])

dice = np.stack([attnunet_dice, convseg_dice, vitseg_dice])
dice = dice - np.min(dice, axis=0, keepdims=True) + 0.01

weights = dice / np.mean(dice, axis=0, keepdims=True)
print(np.round(weights, 3).tolist())

weights = (dice == np.max(dice, axis=0, keepdims=True)).astype(np.float32)
print(weights)