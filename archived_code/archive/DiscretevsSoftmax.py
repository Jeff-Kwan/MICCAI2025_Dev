import timeit
import torch
import monai.transforms as mt

repeats = 1

# Create a random tensor simulating predictions
input_data = {"pred": torch.randn(14, 1, 1, 1)}

# Define transforms
softmax_transform = mt.Activationsd(keys=["pred"], softmax=True)
asdiscrete_transform = mt.AsDiscreted(keys=["pred"], argmax=True, to_onehot=14)

fast_onehot_transform = mt.Compose([
    mt.NormalizeIntensityd(keys=["pred"], subtrahend=0.0, divisor=0.5),  # Sharpen confidence
    mt.Activationsd(keys=["pred"], softmax=True),   # Logits to probabilities
    mt.ThresholdIntensityd(keys=["pred"], above=True, threshold=1/14, cval=0),
])

def bench(transform, data, repeats=repeats):
    # warm-up
    for _ in range(3):
        transform(data)
    # timed runs
    t = timeit.timeit(lambda: transform(data), number=repeats)
    return t / repeats

print(fast_onehot_transform(input_data)["pred"].squeeze())
print(asdiscrete_transform(input_data)["pred"].squeeze())
exit()
# assert torch.all(fast_onehot_transform(input_data)["pred"] == 
#     asdiscrete_transform(input_data)["pred"]), "FastOneHot should match AsDiscreted output"

print("Softmax avg:", bench(softmax_transform, input_data))
print("AsDiscreted avg:", bench(asdiscrete_transform, input_data))
print("FastOneHot avg:", bench(fast_onehot_transform, input_data))
