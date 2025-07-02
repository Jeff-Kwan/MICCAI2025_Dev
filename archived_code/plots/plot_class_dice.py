import json
import numpy as np
import matplotlib.pyplot as plt

data = json.load(open("archived_code\plots\class_dice.json"))
class_dice = data["class_dice"]
class_names = [
    "Liver",
    "Right kidney",
    "Spleen",
    "Pancreas",
    "Aorta",
    "Inferior Vena Cava",
    "Right Adrenal Gland",
    "Left Adrenal Gland",
    "Gallbladder",
    "Esophagus",
    "Stomach",
    "Duodenum",
    "Left kidney"
]
plt.figure(figsize=(12, 6))
class_dice = np.array(class_dice).transpose().tolist()
for name, dice in zip(class_names, class_dice):
    plt.plot(dice, label=name)
plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("Dice Score for Each Organ over Training")
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.grid()
plt.tight_layout()
plt.show()